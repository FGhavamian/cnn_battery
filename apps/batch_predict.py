import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib.pyplot as plt

from trainer.predict import Predictor
from trainer.utils.util import *
from trainer.names import GRID_DIM


PATH_JOB = 'output/data_size/1.0/0.001/hydra_v01/boundary_edge_surface/cnn_20190819_092340'
PATH_MODEL = PATH_JOB + '/model.h5'
PATH_DATA_PROCESSED = 'data/processed/boundary_edge_surface'
PATH_DATA_RAW = 'data/raw/ex2'


def read_data():
    names = read_json(os.path.join(PATH_DATA_PROCESSED, 'names.json'))

    return names['train'], names['test']


def get_grid(pp):
    grid_x = pp.grid['grid'][:, 0].reshape((GRID_DIM.y, GRID_DIM.x))
    grid_y = pp.grid['grid'][:, 1].reshape((GRID_DIM.y, GRID_DIM.x))
    return grid_x, grid_y


def plot(y, pred, grid_x, grid_y, field_name, case_idx):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
    vmin = y[case_idx, :, :].min()
    vmax = y[case_idx, :, :].max()
    pred_some = pred[case_idx, :, :]
    y_some = y[case_idx, :, :]
    ax0_cont = ax0.contourf(grid_x, grid_y, pred_some, 50, vmin=vmin, vmax=vmax)
    ax1_cont = ax1.contourf(grid_x, grid_y, y_some, 50, vmin=vmin, vmax=vmax)
    fig.colorbar(ax0_cont, ax=ax0)
    fig.colorbar(ax1_cont, ax=ax1)
    fig.suptitle(field_name)
    plt.tight_layout(10)


if __name__ == '__main__':
    # read input and target data
    names_train, names_test = read_data()

    predict = Predictor(
        path_model=PATH_MODEL,
        path_stats=PATH_DATA_PROCESSED,
        preprocess_path=PATH_JOB + '/pp.pickle'
    )

    def batch_predict(names, pre_mode):
        paths_mesh = {name: os.path.join(PATH_DATA_RAW, 'mesh', name + '.mesh') for name in names}
        paths_vtu = {name: os.path.join(PATH_DATA_RAW, 'vtu', name + '.vtu') for name in names}

        fields, sig_vm, i_m, c, pp = predict(names, paths_mesh, paths_vtu, pre_mode)
        grid_x, grid_y = get_grid(pp)

        for case_idx in range(len(names[-5:])):
            plot(sig_vm['y'], sig_vm['pred'], grid_x, grid_y, 'von mises stress', case_idx=case_idx)
            plot(i_m['y'], i_m['pred'], grid_x, grid_y, 'electric density intensity', case_idx=case_idx)
            plot(c['y'], c['pred'], grid_x, grid_y, 'concentration', case_idx=case_idx)

        plt.show()

    batch_predict(names_test, pre_mode='test')
