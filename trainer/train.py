import argparse
import json
import os

import tensorflow as tf

from trainer.models import build_model
from trainer.data import make_dataset
from trainer.names import FEATURE_TO_DIM
from trainer.utils.callbacks import get_callbacks
from trainer.utils import make_metrics


def train(args):
    dataset_train = make_dataset(
        args.path_tfrecords,
        batch_size=args.batch_size, mode='train')

    dataset_test = make_dataset(
        args.path_tfrecords,
        batch_size=args.batch_size, mode='test')

    feature_dim = sum(
        [FEATURE_TO_DIM[f] for f in args.feature_name.split('_')]
    )
    model = build_model(
        name=args.model_name,
        feature_dim=feature_dim,
        head_type=args.head_type,
        filters=args.filters,
        kernels=args.kernels)
    
    model.compile(
        tf.keras.optimizers.Adam(lr=args.learning_rate),
        loss='mse',
        metrics=[make_metrics(args.head_type)])

    model.fit(
        x=dataset_train,
        epochs=args.epoch_num,
        validation_data=dataset_test,
        steps_per_epoch=max(1, args.n_train_samples//args.batch_size),
        validation_steps=max(1, args.n_test_samples//args.batch_size),
        verbose=0,
        callbacks=get_callbacks(
            monitor='val_loss',
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
        required=True)

    parser.add_argument(
        '--filters',
        help='underscored seperated list of filters per layer',
        type=lambda x: [int(x_) for x_ in x.split('_')],
        required=True)

    parser.add_argument(
        '--kernels',
        help='list of kernels per layer',
        type=lambda x: [(int(x_), int(x_)) for x_ in x.split('_')],
        required=True)

    parser.add_argument(
        '--feature-name',
        help='name of features',
        type=lambda x: '_'.join(sorted(x.split('_'))),
        required=True)

    parser.add_argument(
        '--path-tfrecords',
        help='path to tfrecords files',
        required=True)

    parser.add_argument(
        '--n-train-samples',
        help='number of training samples',
        type=int,
        required=True)

    parser.add_argument(
        '--n-test-samples',
        help='number of test samples',
        type=int,
        required=True)

    parser.add_argument(
        '--head-type',
        help='choose among "de", "vector", "scalar"',
        default='vector')

    parser.add_argument(
        '--epoch-num',
        help='max number of epochs',
        type=int,
        default=1000)

    parser.add_argument(
        '--learning-rate',
        help='rate of learning',
        type=float,
        default=1e-3)

    parser.add_argument(
        '--ex-path',
        help='example number which is used in the path of output file',
        default='test')

    parser.add_argument(
        '--batch-size',
        help='batch size',
        type=int,
        default=16)

    args = parser.parse_args()

    args.path_output = os.path.join(args.ex_path, args.job_name)
    make_output_directory(args.path_output)

    write_hparams_to_file(args)

    print(f'[INFO] training example at {args.path_output}')
    train(args)
