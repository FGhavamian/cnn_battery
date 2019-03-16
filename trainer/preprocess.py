import argparse

from scipy.interpolate import NearestNDInterpolator

from trainer.util import *
from trainer.features import FeatureMaker
from trainer.names import *


def get_train_val_test_paths(case_dir):
    paths_mesh = glob.glob(os.path.join(case_dir, "mesh", "*.mesh"))
    paths_vtu = glob.glob(os.path.join(case_dir, "vtu", "*.vtu"))

    names = [path.split(os.sep)[-1].replace(".mesh", "") for path in paths_mesh]

    from sklearn.model_selection import train_test_split
    names_train, names_test = train_test_split(names, test_size=0.1, random_state=123)
    # names_train, names_val = train_test_split(names_train, test_size=0.1, random_state=123)

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
            groups_node=groups_node,
            groups_element_nodes=groups_element_nodes,
            element_connect=element_connect)

        self.mesh_data[name] = mesh_dict
        self.compiled = False

    def add_to_vtu_data(self, name, coord, solutions):
        """
        Add vtu data.
        Args:
            name (str): name of the case
            coord (NumPy ndarray): the nodal coordinates
            solutions (dict of str: NumPy ndarray): name of solution field to the solution field
        """
        sols = [solutions[name] for name in SOLUTION_DIMS]
        sols = np.concatenate(sols, axis=1)

        vtu_dict = dict(
            coord=coord,
            solutions=sols)

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

        self.grid = self.__get_grid(self.is_train)

        self.__make_mask()

        self.__encode_geometry()
        self.x = self.__standardize('x', self.x)

        if self.vtu_data:
            self.__interpolate_solution()
            self.y = self.__standardize('y', self.y)

        self.compiled = True

    def populate(self, case_names, paths_mesh, paths_vtu=None):
        if not isinstance(case_names, list):
            case_names = [case_names]

        for i, case_name in enumerate(case_names):
            print('reading file: ' + '(' + str(i) + ') ' + case_name, end='\r')
            nodes_coord, groups_node, groups_element_nodes, element_connect = read_mesh(file_path=paths_mesh[case_name])

            self.add_to_mesh_data(case_name, nodes_coord, groups_node, groups_element_nodes, element_connect)
            self.x_names = list(groups_node.keys())

            if paths_vtu:
                nodes_coord_vtu, solutions = read_vtu(file_path=paths_vtu[case_name])

                self.add_to_vtu_data(case_name, nodes_coord_vtu, solutions)
                # self.y_names = list(solutions.keys())

        print()

    def __get_grid(self, is_train):
        if is_train:
            print('making grid')
            return self.__make_grid()
        else:
            return self.__load_grid()

    def __standardize(self, mode, d):
        if self.is_train:
            stats = self.__compute_stats(d, mode=mode)
        else:
            stats = self.__load_stats(mode=mode)

        d = self.__normalize(d, stats)

        return d

    def __encode_geometry(self):
        print('encoding geometry')
        # start_time = time.time()

        case_names = list(self.mesh_data.keys())

        features_batch = []
        for case_name in case_names:
            feature_maker = FeatureMaker(
                self.grid["grid"],
                self.mesh_data[case_name],
                feature_to_include=self.feature_name
            )

            features = feature_maker()

            features_batch.append(features)

        self.x = np.stack(features_batch)
        self.x = self.x.reshape(-1, *self.grid["dim"], self.x.shape[-1])

        # end_time = time.time()
        # print(end_time - start_time)

    def __interpolate_solution(self):
        print('interpolating solution field on the grid')

        case_names = list(self.vtu_data.keys())
        grid = self.grid["grid"]

        targets_batch = []
        for case_name in case_names:
            node_coord = self.vtu_data[case_name]["coord"]

            target_on_node = self.vtu_data[case_name]["solutions"]

            f = NearestNDInterpolator(node_coord, target_on_node)

            target_on_grid = f(grid)

            targets_batch.append(target_on_grid)

        self.y = np.stack(targets_batch)
        self.y = self.y.reshape(-1, *self.grid["dim"], self.y.shape[-1])

    @staticmethod
    def __normalize(x, stats):
        x = (x - stats["mean"]) / (stats["std"] + 1e-8)

        return x

    def __compute_stats(self, x, mode):
        mean_s = np.mean(x, axis=(0, 1, 2), keepdims=True, dtype=np.float32)
        std_s = np.std(x, axis=(0, 1, 2), keepdims=True, dtype=np.float32)

        stats = {"mean": mean_s, "std": std_s}

        with open(os.path.join(self.path_output, "stats_{}.pkl".format(mode)), 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

        return stats

    def __load_stats(self, mode):
        with open(os.path.join(self.path_output, "stats_{}.pkl".format(mode)), 'rb') as f:
            stats = pickle.load(f)

        # check stats type
        if (stats['mean'].dtype != 'float32') or (stats['std'].dtype != 'float32'):
            raise Exception('dtypes of mean and std are not float32!')

        return stats

    def __make_mask(self):
        print('making masks')

        case_names = list(self.mesh_data.keys())

        masks = []
        for case_name in case_names:
            coord = self.mesh_data[case_name]['coord']

            x_max, y_max = np.array(coord).max(axis=0)

            mask = (self.grid["grid"][:, 0] - x_max < 1e-8) * (self.grid["grid"][:, 1] - y_max < 1e-8)

            mask = mask.reshape(*self.grid["dim"], -1)
            mask = to_float32(mask)

            masks.append(mask)

        self.mask = np.stack(masks)

    def __make_grid(self):
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

    def __load_grid(self):
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

            with tf.python_io.TFRecordWriter(tf_filename) as tfrecords_writer:
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
        '--features',
        help='name of features to include, underscored seperated',
        type=lambda x: sorted(x.split('_')),
        required=True
    )

    parser.add_argument(
        '--path-data',
        help='path to mesh and vtu files',
        default=os.path.join('data', 'raw', 'ex2')
    )

    parser.add_argument(
        '--path-output',
        help='path where processed data are stored',
        default=os.path.join('data', 'processed')
    )

    args = parser.parse_args()

    args.path_output = make_output_dir(args.path_output, args.features)

    pp_train = PreprocessBatch(is_train=True, path_output=args.path_output, feature_name=args.features)
    pp_test = PreprocessBatch(is_train=False, path_output=args.path_output, feature_name=args.features)

    print('reading mesh and vtu files')
    names_train, names_test, paths_mesh, paths_vtu = get_train_val_test_paths(args.path_data)

    # names_train = ['Rc_1.0_h_20.2_tac_10.0_tel_6.0', 'Rc_1.0_h_49.3_tac_10.0_tel_18.0']
    # print(names_train)

    write_json(
        file_path=os.path.join(args.path_output, 'names.json'),
        data=dict(train=names_train, test=names_test)
    )

    pp_train.populate(names_train, paths_mesh, paths_vtu)
    pp_test.populate(names_test, paths_mesh, paths_vtu)

    # compile pp classes
    print('compiling preprocess classes')
    pp_train.compile()
    pp_test.compile()

    # save data to file
    print('writing tfrecords files')
    tfrecords = Tfrecords(path_output=args.path_output)

    tfrecords.write(pp_train, mode='train', names=names_train)
    tfrecords.write(pp_test, mode='test', names=names_test)
