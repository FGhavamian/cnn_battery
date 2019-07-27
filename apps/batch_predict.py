import matplotlib.pyplot as plt

from trainer.predict import Predictor
from trainer.utils.util import *

PATH_JOB = 'output/hydra_v0/boundary/cnn_20190727_095347'
PATH_MODEL = PATH_JOB + '/model.h5'
PATH_DATA_PROCESSED = 'data/processed/boundary'
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
        path_stats=PATH_DATA_PROCESSED,
        preprocess_path=PATH_JOB + '/pp.pickle'
    )

    def batch_predict(names, pre_mode):
        paths_mesh = {name: os.path.join(PATH_DATA_RAW, 'mesh', name + '.mesh') for name in names}
        paths_vtu = {name: os.path.join(PATH_DATA_RAW, 'vtu', name + '.vtu') for name in names}

        pred, ref, pp = predict(names, paths_mesh, paths_vtu, pre_mode)
        print(pred.shape)
        print(ref.shape)
        print(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

        from trainer.names import GRID_DIM
        print(pp.grid['dim'], pp.grid['grid'].shape)
        grid_x = pp.grid['grid'][:, 0].reshape((GRID_DIM.y, GRID_DIM.x))
        grid_y = pp.grid['grid'][:, 1].reshape((GRID_DIM.y, GRID_DIM.x))
        print(grid_x.shape, grid_y.shape)

        for i in range(pred.shape[-1]):
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
            vmin = ref[0, :, :, i].min()
            vmax = ref[0, :, :, i].max()
            pred_some = pred[0, :, :, i].reshape(*pred[0, :, :, 0].shape)
            ref_some = ref[0, :, :, i].reshape(*ref[0, :, :, 0].shape)
            ax0_cont = ax0.contourf(grid_x, grid_y, pred_some , 50, vmin=vmin, vmax=vmax)
            ax1_cont = ax1.contourf(grid_x, grid_y, ref_some, 50, vmin=vmin, vmax=vmax)
            fig.colorbar(ax0_cont, ax=ax0)
            fig.colorbar(ax1_cont, ax=ax1)
        plt.show()

    # batch_predict(names_train, pre_mode='train')
    batch_predict(names_test, pre_mode='test')


if __name__ == '__main__':
    main()
