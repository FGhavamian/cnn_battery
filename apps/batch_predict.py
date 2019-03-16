import os

from trainer.predict import Predictor
from trainer.util import *

PATH_MODEL = 'output/hydra_scalar_v0/boundary_edge_surface/cnn_20181228_114040/model.h5'
PATH_DATA_PROCESSED = 'data/processed/boundary_edge_surface'
PATH_DATA_RAW = 'data/raw/ex2'


def read_data():
    names = read_json(os.path.join(PATH_DATA_PROCESSED, 'names.json'))

    return names['train'], names['test']


def main():
    # read input and target data
    names_train, names_test = read_data()

    # make predictions
    predict = Predictor(
        path_model=PATH_MODEL,
        path_stats=PATH_DATA_PROCESSED
    )

    def batch_predict(names, pre_mode):
        paths_mesh = {name: os.path.join(PATH_DATA_RAW, 'mesh', name + '.mesh') for name in names}
        paths_vtu = {name: os.path.join(PATH_DATA_RAW, 'vtu', name + '.vtu') for name in names}

        predict(names, paths_mesh, paths_vtu, pre_mode)

    batch_predict(names_train, pre_mode='train')
    batch_predict(names_test, pre_mode='test')


if __name__ == '__main__':
    main()
