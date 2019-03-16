import pickle
import argparse

import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator

from trainer.preprocess import PreprocessBatch
from trainer.util import *


def extract_case_name(path_mesh):
    return path_mesh.split('/')[-1].replace('.mesh', '')


def extract_model_name(path_model):
    return path_model.split('/')[1]
    # return os.path.join(*path_model.split('/')[-3:-1])


def extract_feature_name(path_stats):
    return path_stats.split('/')[2].split('_')


class Predictor:
    def __init__(self, path_model, path_stats, do_write2vtk=True):
        """

        Args:
            path_model (): path to the keras model file
            path_stats (): path to the directory where stats_x, stats_y can be found
            path_data (): path to the data directory where .mesh and .vtu file can be found
            do_write2vtk (): do write the outputs to a vtu file?
        """

        self.feature_name = extract_feature_name(path_stats)
        self.model_name = extract_model_name(path_model)

        self.model = self.__load_model(path_model)
        self.stats = self.__load_stats(path_stats)
        self.pp = self.__get_preprocessor(path_stats)

        self.do_write2vtk = do_write2vtk

    def __call__(self, names, paths_mesh, paths_vtu=None, pre_mode=''):
        """

        Args:
            paths_mesh ():
            paths_vtu (): if path_vtu is provided, then the model is evaluated and error is saved in a vtu file

        Returns:

        """
        self.__compile_preprocessor(names, paths_mesh, paths_vtu)

        inputs = self.__get_inputs()

        pred = self.__make_prediction(inputs)
        pred = self.__destandardize_prediction(pred)
        pred = self.__mask_pred(pred, inputs['mask'])
        pred = self.__to_nodes(pred)

        if self.do_write2vtk and paths_vtu:
            ref = self.__destandardize_prediction(self.pp.y)
            ref = self.__to_nodes(ref)

            def err_func(a, b):
                return np.abs(a - b) / np.linalg.norm(b, axis=0, keepdims=True)

            err = {name: err_func(pred[name], ref[name]) for name in names}

            for name in names:
                self.__write2vtk(pred[name], case_name=name, mode='pred', pre_mode=pre_mode)

                if paths_vtu:
                    self.__write2vtk(ref[name], case_name=name, mode='ref', pre_mode=pre_mode)
                    self.__write2vtk(err[name], case_name=name, mode='err', pre_mode=pre_mode)

        return pred

    def __write2vtk(self, pred, case_name, mode, pre_mode):
        node_coord = self.pp.mesh_data[case_name]['coord']
        element_connect = self.pp.mesh_data[case_name]['element_connect']

        feature_name = '_'.join(self.feature_name)

        report_dir = os.path.join('report', self.model_name, feature_name, pre_mode + '_' + case_name)

        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        write_vtu = VtuWriter(
            report_dir=report_dir,
            file_name=mode + '_' + case_name,
            coord=node_coord,
            connect=element_connect,
            pred=pred
        )

        write_vtu()

    def __to_nodes(self, pred_on_grid):
        preds_on_grid = np.vsplit(pred_on_grid, pred_on_grid.shape[0])

        case_names = list(self.pp.mesh_data.keys())
        pred_on_node = {case_name: None for case_name in case_names}

        for case_idx, pred_on_grid in enumerate(preds_on_grid):
            case_name = case_names[case_idx]
            node_coord = self.pp.mesh_data[case_name]['coord']

            pred_on_grid = np.reshape(pred_on_grid, [-1, pred_on_grid.shape[-1]])

            f = NearestNDInterpolator(self.pp.grid['grid'], pred_on_grid)

            pred_on_node[case_name] = f(node_coord)

        return pred_on_node

    def __make_prediction(self, inputs):
        output = self.model.predict(inputs, steps=1)

        return output

    def __get_inputs(self):
        return dict(feature=self.pp.x, mask=self.pp.mask)

    def __compile_preprocessor(self, names, paths_mesh, paths_vtu):
        self.pp.populate(names, paths_mesh, paths_vtu)
        self.pp.compile()

    def __get_preprocessor(self, path_stats):
        return PreprocessBatch(is_train=False, path_output=path_stats, feature_name=self.feature_name)

    @staticmethod
    def __load_model(path_model):
        return keras.models.load_model(
            filepath=path_model,
            custom_objects={func.__name__: func for func in make_metrics()})

    @staticmethod
    def __load_stats(path_data):
        with open(os.path.join(path_data, 'stats_y.pkl'), 'rb') as file:
            return pickle.load(file)
        # return np.load(os.path.join(path_data, 'stats_y.npy'))

    def __destandardize_prediction(self, pred):
        return pred * self.stats["std"] + self.stats["mean"]

    @staticmethod
    def __mask_pred(pred, mask):
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
