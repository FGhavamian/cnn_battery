from scipy.interpolate import NearestNDInterpolator

from trainer.preprocess import PreprocessBatch
from trainer.utils.util import *


# def extract_case_name(path_mesh):
#     return path_mesh.split(os.path.sep)[-1].replace('.mesh', '')


def extract_model_name(path_model):
    return path_model.split(os.path.sep)[1]
    # return os.path.join(*path_model.split('/')[-3:-1])


def extract_feature_name(path_stats):
    return path_stats.split(os.path.sep)[-1].split('_')[:-1]


class Predictor:
    def __init__(self, path_model, path_stats, preprocess_path, do_write2vtk=True):
        """

        Args:
            path_model (): path to the keras model file
            path_stats (): path to the directory where stats_x, stats_y can be found
            preprocess_path (): path at which the preprocessor is saved as a pickle file
            do_write2vtk (): do write the outputs to a vtu file?
        """
        self.feature_name = extract_feature_name(path_stats)
        # self.model_name = extract_model_name(path_model)

        self.model = self._load_model(path_model)
        self.stats = self._load_stats(path_stats)
        self.pp = self._get_preprocessor(path_stats)

        self.do_write2vtk = do_write2vtk
        self.preprocess_path = preprocess_path

    def __call__(self, names, paths_mesh, paths_vtu=None, pre_mode=''):
        self._compile_preprocessor(names, paths_mesh, paths_vtu)

        inputs = self._get_inputs()

        pred = self._make_prediction(inputs)
        pred = self._destandardize_prediction(pred)
        pred = self._mask_pred(pred, inputs['mask'])

        y = self._destandardize_prediction(self.pp.y)

        fields_geom = inputs

        sig_vm = {
            'y': self._compute_vm_stress(y),
            'pred': self._compute_vm_stress(pred)
        }

        i_m = {
            'y': self._compute_elec_current_density_intensity(y),
            'pred': self._compute_elec_current_density_intensity(pred)
        }

        c = {
            'y': self._compute_ion_concentration(y),
            'pred': self._compute_ion_concentration(pred)
        }

        fields_sol = {
            'y': y,
            'pred': pred
        }

        return fields_geom, fields_sol, sig_vm, i_m, c, self.pp

    @staticmethod
    def _compute_vm_stress(fields):
        sig_xx = fields[:, :, :, 3]
        sig_yy = fields[:, :, :, 4]
        sig_zz = fields[:, :, :, 5]
        sig_xy = fields[:, :, :, 6]

        sig_vm = np.sqrt(0.5*((sig_xx-sig_yy)**2 + (sig_xx-sig_zz)**2 + (sig_yy-sig_zz)**2 + 6*sig_xy**2))
        return sig_vm

    @staticmethod
    def _compute_elec_current_density_intensity(fields):
        i_x = fields[:, :, :, 8]
        i_y = fields[:, :, :, 9]

        i_m = np.sqrt(i_x**2 + i_y**2)
        return i_m

    @staticmethod
    def _compute_ion_concentration(fields):
        c = fields[:, :, :, -3]
        return c

        # pred = self.__to_nodes(pred)
        #
        # if self.do_write2vtk and paths_vtu:
        #     ref = self.__destandardize_prediction(self.pp.y)
        #     ref = self.__to_nodes(ref)
        #
        #     def err_func(a, b):
        #         return np.abs(a - b) / np.linalg.norm(b, axis=0, keepdims=True)
        #
        #     err = {name: err_func(pred[name], ref[name]) for name in names}
        #
        #     for name in names:
        #         self.__write2vtk(pred[name], case_name=name, mode='pred', pre_mode=pre_mode)
        #
        #         if paths_vtu:
        #             self.__write2vtk(ref[name], case_name=name, mode='ref', pre_mode=pre_mode)
        #             self.__write2vtk(err[name], case_name=name, mode='err', pre_mode=pre_mode)

    # def _write2vtk(self, pred, case_name, mode, pre_mode):
    #     node_coord = self.pp.mesh_data[case_name]['coord']
    #     element_connect = self.pp.mesh_data[case_name]['element_connect']
    #
    #     feature_name = '_'.join(self.feature_name)
    #
    #     report_dir = os.path.join('report', self.model_name, feature_name, pre_mode + '_' + case_name)
    #
    #     if not os.path.exists(report_dir):
    #         os.makedirs(report_dir)
    #
    #     write_vtu = VtuWriter(
    #         report_dir=report_dir,
    #         file_name=mode + '_' + case_name,
    #         coord=node_coord,
    #         connect=element_connect,
    #         pred=pred
    #     )
    #
    #     write_vtu()

    # def _to_nodes(self, pred_on_grid):
    #     preds_on_grid = np.vsplit(pred_on_grid, pred_on_grid.shape[0])
    #
    #     case_names = list(self.pp.mesh_data.keys())
    #     pred_on_node = {case_name: None for case_name in case_names}
    #
    #     for case_idx, pred_on_grid in enumerate(preds_on_grid):
    #         case_name = case_names[case_idx]
    #         node_coord = self.pp.mesh_data[case_name]['coord']
    #
    #         pred_on_grid = np.reshape(pred_on_grid, [-1, pred_on_grid.shape[-1]])
    #
    #         f = NearestNDInterpolator(self.pp.grid['grid'], pred_on_grid)
    #
    #         pred_on_node[case_name] = f(node_coord)
    #
    #     return pred_on_node

    def _make_prediction(self, inputs):
        return self.model.predict(inputs, steps=1)

    def _get_inputs(self):
        return dict(feature=self.pp.x, mask=self.pp.mask)

    def _compile_preprocessor(self, names, paths_mesh, paths_vtu):
        if os.path.exists(self.preprocess_path):
            with open(self.preprocess_path, 'rb') as file:
                self.pp = pickle.load(file)
        else:
            self.pp.populate(names, paths_mesh, paths_vtu)
            self.pp.compile()
            with open(self.preprocess_path, 'wb') as file:
                pickle.dump(self.pp, file)

    def _get_preprocessor(self, path_stats):
        return PreprocessBatch(is_train=False, path_output=path_stats, feature_name=self.feature_name, grid_dim=(512, 64))

    @staticmethod
    def _load_model(path_model):
        return keras.models.load_model(
            filepath=path_model,
            custom_objects={func.__name__: func for func in make_metrics()})

    @staticmethod
    def _load_stats(path_data):
        with open(os.path.join(path_data, 'stats_y.pkl'), 'rb') as file:
            return pickle.load(file)
        # return np.load(os.path.join(path_data, 'stats_y.npy'))

    def _destandardize_prediction(self, pred):
        return pred * self.stats["std"] + self.stats["mean"]

    @staticmethod
    def _mask_pred(pred, mask):
        return pred * mask


# def main(args):
#     predict = Predictor(args.path_model, args.path_data)
#
#     pred = predict(args.path_mesh)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument(
#         '--path-model',
#         help='path to the cnn model file',
#         required=True
#     )
#
#     parser.add_argument(
#         '--path-stats',
#         help='path to the directory where stats_x, stats_y can be found',
#         required=True
#     )
#
#     parser.add_argument(
#         '--path-data',
#         help='path to the data directory',
#         required=True
#     )
#
#     parser.add_argument(
#         '--case-name',
#         help='name on mesh (and vtu) file',
#         required=True
#     )
#
#     parser.add_argument(
#         '--do-evaluate',
#         help='do read vtu file and compute error?',
#         type=int,
#         required=True
#     )
#
#     args = parser.parse_args()
#
#     main(args)

    # node_coord = np.array([
    #     [0, 0],
    #     [2, 0],
    #     [2, 2],
    #     [0, 2],
    #     [1, 1]
    # ])
    #
    # connect = [
    #     [0, 1, 4],
    #     [1, 2, 4],
    #     [4, 2, 3],
    #     [0, 4, 3]
    # ]
    # connect = dict(zip([0, 1, 2, 3], connect))
    #
    # pred = np.array([[-1, -1, 3, 3, 1]]).T
    # pred = np.repeat(pred, 17, axis=1)
    #
    # write_vtu = VtuWriter(
    #     model_name='cnn_test',
    #     case_name='0',
    #     coord=node_coord,
    #     connect=connect,
    #     pred=pred
    # )
    #
    # write_vtu()
