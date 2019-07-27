import argparse
import json
import os

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam

from trainer.model import get_model
from trainer.data import make_dataset
from trainer.names import FEATURE_TO_DIM
from trainer.utils.callbacks import PrettyLogger
from trainer.utils import make_metrics


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

    pl = PrettyLogger(display=5)

    return [chp, tb, rlr, pl]


def train(args):
    dataset_train = make_dataset(
        args.path_tfrecords,
        batch_size=16, mode='train')

    dataset_test = make_dataset(
        args.path_tfrecords,
        batch_size=16, mode='test')

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
        steps_per_epoch=10,
        validation_steps=10,
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
        default=2000
    )

    parser.add_argument(
        '--learning-rate',
        help='rate of learning',
        type=float,
        default=1e-3)

    args = parser.parse_args()

    args.path_output = os.path.join('output', args.model_name, args.feature_name, args.job_name)
    make_output_directory(args.path_output)

    write_hparams_to_file(args)

    train(args)
