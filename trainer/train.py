import argparse
import json
import os

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam

from trainer.model import get_model
from trainer.data import make_dataset
from trainer.names import FEATURE_TO_DIM
from trainer.utils.callbacks import PrettyLogger
from trainer.utils import make_metrics

TRAIN_NUM = 40
TEST_NUM = 10


def get_callbacks(monitor, mode, args):
    chp = ModelCheckpoint(
        filepath=os.path.join(args.path_output, "model.h5"),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        period=args.epoch_num // 100,
        verbose=1)

    # es = EarlyStopping(
    #     monitor=monitor,
    #     patience=args.epoch_num // 1000,
    #     min_delta=1e-5,
    #     mode=mode,
    #     verbose=1)

    tb = TensorBoard(
        log_dir=os.path.join(args.path_output, "graph"),
        histogram_freq=0,
        write_graph=True,
        write_grads=False)

    rlr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=args.epoch_num // 10,
        min_lr=1e-5,
        mode=mode,
        min_delta=1e-2,
        verbose=1)

    pl = PrettyLogger(display=20)

    return [chp, tb, rlr, pl]


def train(args):
    dataset_train = make_dataset(
        args.path_tfrecords,
        batch_size=args.batch_size, mode='train')

    dataset_test = make_dataset(
        args.path_tfrecords,
        batch_size=args.batch_size, mode='test')

    feature_dim = sum([FEATURE_TO_DIM[f] for f in args.feature_name.split('_')])

    model = get_model(args.model_name, feature_dim=feature_dim)

    model.compile(
        Adam(lr=args.learning_rate),
        loss='mse',
        metrics=make_metrics())

    model.fit(
        x=dataset_train,
        epochs=args.epoch_num,
        validation_data=dataset_test,
        steps_per_epoch=max(1, TRAIN_NUM//args.batch_size),
        validation_steps=max(1, TEST_NUM//args.batch_size),
        verbose=0,
        callbacks=get_callbacks(
            monitor='loss',
            mode='min',
            args=args))


def write_hparams_to_file(args):
    with open(os.path.join(args.path_output, "hparams.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def make_output_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name',
        help='name of the cnn model',
        required=True)

    parser.add_argument(
        '--job-name',
        help='name of job',
        required=True
    )

    parser.add_argument(
        '--feature-name',
        help='name of features',
        type=lambda x: '_'.join(sorted(x.split('_'))),
        required=True
    )

    parser.add_argument(
        '--path-tfrecords',
        help='path to tfrecords files',
        required=True
    )

    parser.add_argument(
        '--epoch-num',
        help='max number of epochs',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--learning-rate',
        help='rate of learning',
        type=float,
        default=1e-3)

    parser.add_argument(
        '--ex-path',
        help='example number which is used in the path of output file',
        default='test'
    )

    parser.add_argument(
        '--batch-size',
        help='batch size',
        type=int,
        default=16
    )

    args = parser.parse_args()

    args.path_output = os.path.join(args.ex_path, args.job_name)
    make_output_directory(args.path_output)

    write_hparams_to_file(args)

    print(f'[INFO] training example at {args.path_output}')
    train(args)
