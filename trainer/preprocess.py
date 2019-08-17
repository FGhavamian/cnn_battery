import argparse

from sklearn.model_selection import train_test_split
from scipy.spatial import Delaunay
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from trainer.utils.util import *
from trainer.features import FeatureMaker
from trainer.names import *


def get_train_val_test_paths(case_dir):
    if not os.path.exists(case_dir):
        raise IOError(f'directory {case_dir} does not exist')

    paths_mesh = glob.glob(os.path.join(case_dir, "mesh", "*.mesh"))
    paths_vtu = glob.glob(os.path.join(case_dir, "vtu", "*.vtu"))

    names = [path.split(os.sep)[-1].replace(".mesh", "") for path in paths_mesh]
    names_train, names_test = train_test_split(names, test_size=0.2, random_state=123)

    paths_mesh = {p.split(os.sep)[-1].replace(".mesh", ""): p for p in paths_mesh}
    paths_vtu = {p.split(os.sep)[-1].replace(".vtu", ""): p for p in paths_vtu}

    # return names_train, names_test, names_val, paths_mesh, paths_vtu
    return names_train, names_test, paths_mesh, paths_vtu


class DataHolder:
    def __init__(self):
        self.mesh_data = {}
        self.vtu_data = {}

        self.compiled = False

    def add_to_mesh_data(self, name, coord, groups_node, groups_element_nodes, element_connect):
        """
        Add mesh data.
        Args:
            name (str): name of the case
            coord (NumPy ndarray): the nodal coordinates
            groups_node (dict of str: list): name of node groups to list of nodes
            groups_element_nodes (dict of str: list): name of element groups to list of nodes in that element
            element_connect (dict of int: list): element number to list of nodes
        """
        mesh_dict = dict(
            coord=coord,
            connect=element_connect,
            groups_nodes=groups_node,
            groups_element_nodes=groups_element_nodes)

        self.mesh_data[name] = mesh_dict
        self.compiled = False

    def add_to_vtu_data(self, name, coord, connects, solutions):
        """
        Add vtu data.
        Args:
            name (str): name of the case
            coord (NumPy ndarray): the nodal coordinates
            solutions (dict of str: NumPy ndarray): name of solution field to the solution field
        """
        solutions = {(name, dim): solutions[name][:, dim] for name in SOLUTION_DIMS for dim in range(SOLUTION_DIMS[name])}
        df_sols = pd.DataFrame(solutions)
        # sols = [solutions[name] for name in SOLUTION_DIMS]
        # sols = np.concatenate(sols, axis=1)

        vtu_dict = dict(
            coord=coord,
            connect=connects,
            solutions=df_sols)

        self.vtu_data[name] = vtu_dict
        self.compiled = False


class PreprocessBatch(DataHolder):
    def __init__(self, is_train, path_output, feature_name, grid_size=0.5):
        DataHolder.__init__(self)

        self.is_train = is_train
        self.path_output = path_output
        self.feature_name = feature_name
        self.grid_size = grid_size

        self.grid = None

        self.mask = None

        self.x = None
        self.y = None

        self.x_names = None
        # self.y_names = None

    def compile(self):
        if not self.mesh_data:
            raise (Exception('Set mesh first!'))

        self.mesh_data = self._to_delaunay(self.mesh_data)
        self.vtu_data = self._to_delaunay(self.vtu_data)

        self.grid = self._get_grid(self.is_train)
        self.mask = self._make_mask()

        print('[INFO] encoding geometry ...')
        features = self._encode_geometry()
        targets = {name: data['solutions'] for name, data in self.vtu_data.items()}

        print('[INFO] interpolating features ...')

        self.x = self._interpolate(features, self.grid['grid'], self.mesh_data)
        self.x = self._standardize('x', self.x)

        if self.vtu_data:
            print('[INFO] interpolating solutions ...')
            self.y = self._interpolate(targets, self.grid['grid'], self.vtu_data)
            self.y = self._standardize('y', self.y)

        self.compiled = True

        # import matplotlib.pyplot as plt
        # for i in range(self.y.shape[-1]):
        #     plt.figure()
        #     plt.contourf(
        #         self.grid['grid'][:, 0].reshape(self.grid['dim']),
        #         self.grid['grid'][:, 1].reshape(self.grid['dim']),
        #         self.y[0, :, i].reshape(self.grid['dim']))
        #     plt.show()

    def populate(self, case_names, paths_mesh, paths_vtu=None):
        if not isinstance(case_names, list):
            case_names = [case_names]

        for i, case_name in enumerate(case_names):
            print('reading file: ' + '(' + str(i) + ') ' + case_name, end='\r')
            nodes_coord, groups_node, groups_element_nodes, element_connect = read_mesh(file_path=paths_mesh[case_name])

            self.add_to_mesh_data(case_name, nodes_coord, groups_node, groups_element_nodes, element_connect)
            self.x_names = list(groups_node.keys())

            if paths_vtu:
                nodes_coord_vtu, connects, solutions = read_vtu(file_path=paths_vtu[case_name])

                self.add_to_vtu_data(case_name, nodes_coord_vtu, connects, solutions)
                # self.y_names = list(solutions.keys())

        print()

    @staticmethod
    def _shape_func_tri3(coord, el_coords):
        x1 = el_coords[0][0]; y1 = el_coords[0][1]
        x2 = el_coords[1][0]; y2 = el_coords[1][1]
        x3 = el_coords[2][0]; y3 = el_coords[2][1]

        a0 = x1; b0 = y1
        a1 = x2 - x1; b1 = y2 - y1
        a2 = x3 - x1; b2 = y3 - y1

        xhat = coord[0]; yhat = coord[1]

        etahat = (a1 * yhat - b1 * xhat + b1 * a0 - a1 * b0) / (a1 * b2 - a2 * b1 + 1e-8)
        xihat = (xhat - a0 - a2 * etahat) / (a1 + 1e-8)

        n1 = 1 - xihat - etahat
        n2 = xihat
        n3 = etahat

        n = np.array([n1, n2, n3])

        return n

    @staticmethod
    def _find_grids_in_each_element(grid, mesh):
        point_to_element = mesh.find_simplex(grid, bruteforce=True, tol=1e-1)
        element_to_point = [None for e in range(len(mesh.simplices))]

        for eid in range(len(element_to_point)):
            element_to_point[eid] = np.where(point_to_element == eid)[0]

        return element_to_point

    def _interpolate(self, fields_dict, grid, mesh_data):
        fields_on_grid = []

        def interpolator(case_name):
            mesh = mesh_data[case_name]['mesh']
            fields = fields_dict[case_name]

            # find elements containing grid points
            element_to_point = self._find_grids_in_each_element(grid, mesh)
            element_to_point_some = {e: ps for e, ps in enumerate(element_to_point) if len(ps) > 0}

            # interpolate solutions
            grid_fields = [np.zeros([fields.shape[1]]) for _ in range(len(grid))]

            for e, grid_nums in element_to_point_some.items():
                el_nodes = mesh.simplices[e]
                el_coords = mesh.points[el_nodes]
                el_fields = fields.iloc[el_nodes].values

                for grid_num in grid_nums:
                    grid_coord = grid[grid_num]
                    n = self._shape_func_tri3(grid_coord, el_coords)
                    grid_field = n.dot(el_fields)

                    grid_fields[grid_num] = grid_field

            grid_fields = np.stack(grid_fields, axis=0)
            grid_fields = grid_fields.reshape(*self.grid['dim'] + (-1,))
            return grid_fields

        fields_on_grid = Parallel(n_jobs=8, prefer="threads")(delayed(interpolator)(case_name)
                                                              for case_name in tqdm(fields_dict.keys()))
        fields_on_grid = np.stack(fields_on_grid, axis=0)
        fields_on_grid = fields_on_grid.astype(np.float32)

        return fields_on_grid

    @staticmethod
    def _to_delaunay(data_dict):
        for name, data in data_dict.items():
            mesh = Delaunay(data['coord'])
            mesh.simplices = data['connect'].astype(np.int32)
            data_dict[name]['mesh'] = mesh

        return data_dict

    def _get_grid(self, is_train):
        if is_train:
            return self._make_grid()
        else:
            return self._load_grid()

    def _standardize(self, mode, d):
        if self.is_train:
            stats = self._compute_stats(d, mode=mode)
        else:
            stats = self._load_stats(mode=mode)

        d = self._normalize(d, stats)

        return d

    def _encode_geometry(self):
        # start_time = time.time()

        case_names = list(self.mesh_data.keys())

        features = {}
        for case_name in tqdm(case_names):
            feature_maker = FeatureMaker(
                self.mesh_data[case_name],
                feature_to_include=self.feature_name
            )

            df_features = feature_maker()

            features[case_name] = df_features

        # end_time = time.time()
        # print(end_time - start_time)

        return features


    # def _interpolate_solution(self):
    #     case_names = list(self.vtu_data.keys())
    #     grid = self.grid["grid"]
    #
    #     targets_batch = []
    #     for case_name in case_names:
    #         node_coord = self.vtu_data[case_name]["coord"]
    #
    #         target_on_node = self.vtu_data[case_name]["solutions"]
    #
    #         f = NearestNDInterpolator(node_coord, target_on_node)
    #
    #         target_on_grid = f(grid)
    #
    #         targets_batch.append(target_on_grid)
    #
    #     self.y = np.stack(targets_batch)
    #     self.y = self.y.reshape(-1, *self.grid["dim"], self.y.shape[-1])

    @staticmethod
    def _normalize(x, stats):
        x = (x - stats["mean"]) / (stats["std"] + 1e-8)

        return x

    def _compute_stats(self, x, mode):
        mean_s = np.mean(x, axis=(0, 1, 2), keepdims=True, dtype=np.float32)
        std_s = np.std(x, axis=(0, 1, 2), keepdims=True, dtype=np.float32)

        stats = {"mean": mean_s, "std": std_s}

        with open(os.path.join(self.path_output, "stats_{}.pkl".format(mode)), 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

        return stats

    def _load_stats(self, mode):
        with open(os.path.join(self.path_output, "stats_{}.pkl".format(mode)), 'rb') as f:
            stats = pickle.load(f)

        # check stats type
        if (stats['mean'].dtype != 'float32') or (stats['std'].dtype != 'float32'):
            raise Exception('dtypes of mean and std are not float32!')

        return stats

    def _make_mask(self):
        case_names = list(self.mesh_data.keys())

        masks = []
        for case_name in case_names:
            coord = self.mesh_data[case_name]['coord']

            x_max, y_max = np.array(coord).max(axis=0)

            mask = (self.grid["grid"][:, 0] - x_max < 1e-8) * (self.grid["grid"][:, 1] - y_max < 1e-8)

            mask = mask.reshape(*self.grid["dim"], -1)
            mask = to_float32(mask)

            masks.append(mask)

        return np.stack(masks)

    def _make_grid(self):
        x_max = 0
        y_max = 0

        for _, mesh in self.mesh_data.items():
            x_y_max = mesh["coord"].max(axis=0)

            if x_y_max[0] > x_max:
                x_max = x_y_max[0]

            if x_y_max[1] > y_max:
                y_max = x_y_max[1]

        # # make grid sizes such that they are powers of two
        # grid_num_x = 2 ** np.ceil(np.log2(X_MAX / self.grid_size))
        # grid_num_y = 2 ** np.ceil(np.log2(Y_MAX / self.grid_size))

        x = np.linspace(0, x_max, GRID_DIM.x, dtype=np.float32)
        y = np.linspace(0, y_max, GRID_DIM.y, dtype=np.float32)

        x_grid, y_grid = np.meshgrid(x, y)

        flatten_grid = np.hstack([x_grid.flatten()[:, None], y_grid.flatten()[:, None]])
        grid_dim = x_grid.shape

        grid = {'grid': flatten_grid, 'dim': grid_dim}

        save_to_pickle(output_path=os.path.join(self.path_output, 'grid.pkl'), obj=grid)

        return grid

    def _load_grid(self):
        return load_from_pickle(path=os.path.join(self.path_output, 'grid.pkl'))


class Tfrecords:
    def __init__(self, path_output):
        self.output_dir = os.path.join(path_output, 'tfrecords')
        self.sample_per_file = 100

    def write(self, pp: PreprocessBatch, names, mode):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        x_chunks = list(chunks(pp.x, self.sample_per_file))
        y_chunks = list(chunks(pp.y, self.sample_per_file))
        mask_chunks = list(chunks(pp.mask, self.sample_per_file))
        name_chunks = list(chunks(names, self.sample_per_file))

        for idx in range(len(x_chunks)):
            tf_filename = os.path.join(self.output_dir, mode + '_' + str(idx) + '.tfrecords')

            x_chunk = x_chunks[idx]
            y_chunk = y_chunks[idx]
            mask_chunk = mask_chunks[idx]
            name_chunk = name_chunks[idx]

            with tf.io.TFRecordWriter(tf_filename) as tfrecords_writer:
                for idx_case in range(x_chunk.shape[0]):
                    x = x_chunk[idx_case]
                    y = y_chunk[idx_case]
                    mask = mask_chunk[idx_case]
                    name = name_chunk[idx_case]
                    self.add_to_tfrecords(x, y, mask, name, tfrecords_writer)

    def add_to_tfrecords(self, x, y, mask, name, tfrecords_writer):
        example = tf.train.Example(
            features=tf.train.Features(
                feature=self.get_feature(
                    x=numpy_to_bytes(x),
                    y=numpy_to_bytes(y),
                    mask=numpy_to_bytes(mask),
                    height=np.int64(x.shape[0]),
                    width=np.int64(x.shape[1]),
                    depth_x=np.int64(x.shape[2]),
                    depth_y=np.int64(y.shape[2]),
                    name=str.encode(name)
                )))

        tfrecords_writer.write(example.SerializeToString())

    @staticmethod
    def get_feature(x, y, mask, height, width, depth_x, depth_y, name):
        return dict(
            x=bytes_feature(x),
            y=bytes_feature(y),
            height=int64_feature(height),
            width=int64_feature(width),
            depth_x=int64_feature(depth_x),
            depth_y=int64_feature(depth_y),
            mask=bytes_feature(mask),
            name=bytes_feature(name))


def make_output_dir(path_output, features):
    path_output = os.path.join(path_output, '_'.join(features))
    if not os.path.exists(path_output):
        os.makedirs(path_output)
        return path_output
    elif (
            os.path.exists(os.path.join(path_output, 'tfrecords', 'train_0.tfrecords')) and
            os.path.exists(os.path.join(path_output, 'tfrecords', 'test_0.tfrecords')) and
            os.path.exists(os.path.join(path_output, 'grid.pkl')) and
            os.path.exists(os.path.join(path_output, 'names.json')) and
            os.path.exists(os.path.join(path_output, 'stats_x.pkl')) and
            os.path.exists(os.path.join(path_output, 'stats_y.pkl'))):

        import sys
        sys.exit('Tfrecords files already exist')
    else:
        return path_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--features', '-f',
        help='name of features to include, underscored seperated',
        type=lambda x: sorted(x.split('_')),
        required=True
    )

    parser.add_argument(
        '--path-data', '-d',
        help='path to mesh and vtu files',
        default=os.path.join('data', 'raw', 'ex2')
    )

    parser.add_argument(
        '--path-output', '-o',
        help='path where processed data are stored',
        default=os.path.join('data', 'processed')
    )

    args = parser.parse_args()
    args.path_output = make_output_dir(args.path_output, args.features)

    print('[INFO] reading mesh and vtu files ...')
    names_train, names_test, paths_mesh, paths_vtu = get_train_val_test_paths(args.path_data)

    pp_train = PreprocessBatch(is_train=True, path_output=args.path_output, feature_name=args.features)
    pp_test = PreprocessBatch(is_train=False, path_output=args.path_output, feature_name=args.features)

    # names_train = ['Rc_1.0_h_20.2_tac_10.0_tel_6.0', 'Rc_1.0_h_49.3_tac_10.0_tel_18.0']
    # print(names_train)

    write_json(
        file_path=os.path.join(args.path_output, 'names.json'),
        data=dict(train=names_train, test=names_test)
    )

    pp_train.populate(names_train, paths_mesh, paths_vtu)
    pp_test.populate(names_test, paths_mesh, paths_vtu)

    print('[INFO] compiling preprocess classes ...')
    pp_train.compile()
    pp_test.compile()

    # save data to file
    print('[INFO] writing tfrecords files ...')
    tfrecords = Tfrecords(path_output=args.path_output)

    tfrecords.write(pp_train, mode='train', names=names_train)
    tfrecords.write(pp_test, mode='test', names=names_test)
