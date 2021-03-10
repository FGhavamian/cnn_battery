import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from trainer.predict import Predictor
from trainer.utils.util import *
from trainer.names import GRID_DIM


# learning rate
PATH_JOB = os.path.join(
    'output',
    'data_percentage',
    '1.0',
    '0.001', 
    'simple',
    'boundary_edge_surface',
    'filter_32_64',
    'kernel_7',
    'cnn_20201220_113015' 
)

PATH_DATA_PROCESSED = os.path.join(
    'data',
    'processed',
    f'{PATH_JOB.split(os.path.sep)[-4]}_1.0'
)

PATH_MODEL = os.path.join(PATH_JOB, 'model.h5')
PATH_DATA_RAW = os.path.join('data', 'raw')
PATH_PLOTS = os.path.join(PATH_JOB, 'plots')


def read_data():
    names = read_json(os.path.join(PATH_DATA_PROCESSED, 'names.json'))
    return names['train'], names['test']


def get_grid(pp):
    grid_x = pp.grid['grid'][:, 0].reshape((GRID_DIM.y, GRID_DIM.x))
    grid_y = pp.grid['grid'][:, 1].reshape((GRID_DIM.y, GRID_DIM.x))
    return grid_x, grid_y


def plot(y, pred, grid_x, grid_y, field_name, case_idx, case_name):
    vmin = y[case_idx, :, :].min()
    vmax = y[case_idx, :, :].max()
    pred_some = pred[case_idx, :, :]
    y_some = y[case_idx, :, :]
    err = np.abs(pred_some - y_some)/(np.linalg.norm(y_some) + 1e-8)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 7))
    cbar_ax = fig.add_axes([1.05, 0.1, 0.04, 0.8])

    ax0_cont = ax0.contourf(grid_x, grid_y, pred_some, 50, vmin=vmin, vmax=vmax)
    ax1_cont = ax1.contourf(grid_x, grid_y, y_some, 50, vmin=vmin, vmax=vmax)
    # ax2_cont = ax2.contourf(grid_x, grid_y, err + 1e-15, 50, norm=LogNorm())

    ax0.set_title('HydraNet prediction')
    ax1.set_title('FEM reference')
    # ax2.set_title('error = |CNN - FEM| / ||CNN||_2')

    plt.colorbar(ax1_cont, cax=cbar_ax)
    # fig.colorbar(ax0_cont, ax=ax0)
    # fig.colorbar(ax1_cont, ax=ax1)
    # fig.colorbar(ax2_cont, ax=ax2)
    # fig.suptitle(field_name + '\n' + case_name)
    plt.tight_layout(pad=3)
    plt.suptitle(field_name)

    if not os.path.exists(PATH_PLOTS):
        os.mkdir(PATH_PLOTS)

    plot_path = os.path.join(PATH_PLOTS, field_name + "_" + case_name + ".png")
    plt.savefig(plot_path, format="png")


def plot_fields(geom_fields, grid_x, grid_y):
    vmin = 0
    vmax = 1

    for field_idx in range(geom_fields.shape[-1]):
        fig, ax = plt.subplots(figsize=(4, 7))
        ax_cont = ax.contourf(grid_x, grid_y, geom_fields[:, :, field_idx], 50, vmin=vmin, vmax=vmax)

        # ax.set_title('CNN prediction')

        fig.colorbar(ax_cont, ax=ax)
    # fig.suptitle(field_name + '\n' + case_name)
    # plt.tight_layout(5)


def batch_predict(names, pre_mode):
    paths_mesh = {name: os.path.join(PATH_DATA_RAW, 'mesh', name + '.mesh') for name in names}
    paths_vtu = {name: os.path.join(PATH_DATA_RAW, 'vtu', name + '.vtu') for name in names}

    time_start = time.time()
    geom_fields, fields, sig_vm, i_m, c, pp = predict(names, paths_mesh, paths_vtu, pre_mode)
    time_end = time.time()
    print(f'Execution time per specimen: {(time_end - time_start)/len(names)} seconds')

    grid_x, grid_y = get_grid(pp)

    for case_idx, name in enumerate(names):
        plot(sig_vm['y'], sig_vm['pred'], grid_x, grid_y, 'von Mises stress', case_idx=case_idx, case_name=name)
        plot(i_m['y'], i_m['pred'], grid_x, grid_y, 'Electric density intensity', case_idx=case_idx, case_name=name)
        plot(c['y'], c['pred'], grid_x, grid_y, 'Concentration', case_idx=case_idx, case_name=name)

    # for idx in range(1):
    #     plot_fields(geom_fields['feature'][idx], grid_x, grid_y)
    plt.show()


if __name__ == '__main__':
    predict = Predictor(
        path_model=PATH_MODEL,
        path_stats=PATH_DATA_PROCESSED,
        preprocess_path=os.path.join(PATH_JOB, 'pp.pickle')
    )

    names_train, names_test = read_data()

    batch_predict(names_test, pre_mode='test')
