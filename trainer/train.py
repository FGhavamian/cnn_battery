import argparse
import json

from tensorflow import keras

from trainer.model import *
from trainer.data import *
from trainer.names import *


class PrettyLogger(keras.callbacks.Callback):
    def __init__(self, display):
        super().__init__()
        self.display = display
        self.logs_old = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.display == 0:
            self.print_all(epoch, logs)
        else:
            self.print_some(epoch, logs)

    def print_all(self, epoch, logs):
        print('\n\n{}/{}'.format(epoch, self.params['epochs']))

        if not self.logs_old:
            self.logs_old = logs

        # metrics = [m for m in self.params['metrics'] if 'val' not in m]
        # for metric in metrics:
        #     print('\t{:<15} {:<15.3f} -> {:<20.3f} {:<20} {:<15.3f} -> {:<20.3f}'.format(
        #         metric, self.logs_old[metric], logs[metric],
        #         'val_' + metric, self.logs_old['val_' + metric], logs['val_' + metric]
        #     ))

        for metric in self.params['metrics']:
            print('\t{:<15} {:<15.3f} -> {:<20.3f}'.format(
                metric, self.logs_old[metric], logs[metric])
            )

        self.logs_old = logs

        print()

    def print_some(self, epoch, logs):
        print('{}/{} -- loss {}'.format(epoch, self.params['epochs'], logs['loss']), end='\r')


def get_callbacks(monitor, mode, args):
    chp = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.path_output, "model.h5"),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        period=args.epoch_num // 100,
        verbose=1)

    es = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=args.epoch_num // 1000,
        min_delta=1e-5,
        mode=mode,
        verbose=1)

    tb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.path_output, "graph"),
        histogram_freq=0,
        write_graph=True,
        write_grads=False)

    rlr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=args.learning_rate // 2000,
        min_lr=1e-5,
        mode=mode,
        min_delta=1e-4,
        verbose=1)

    return [chp, tb, rlr, PrettyLogger(display=5)]


def train(args):
    dataset_train = make_dataset(
        args.path_tfrecords,
        batch_size=128, mode='train')

    feature_dim = sum([FEATURE_TO_DIM[f] for f in args.feature_name.split('_')])

    model = get_model(args.model_name, feature_dim=feature_dim)

    model.compile(
        keras.optimizers.Adam(lr=args.learning_rate),
        loss='mse',
        metrics=make_metrics())

    model.fit(
        x=dataset_train,
        epochs=args.epoch_num,
        # validation_data=dataset_val,
        steps_per_epoch=1,
        # validation_steps=1,
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
        default=1e-4)

    args = parser.parse_args()

    args.path_output = os.path.join('output', args.model_name, args.feature_name, args.job_name)
    make_output_directory(args.path_output)

    write_hparams_to_file(args)

    train(args)
