import os

import tensorflow as tf


class PrettyLogger(tf.keras.callbacks.Callback):
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

        for metric in logs.keys():
            print('\t{:<30} {:<5.3f} -> {:<10.3f}'.format(
                metric, self.logs_old[metric], logs[metric]))

        self.logs_old = logs

        print()

    def print_some(self, epoch, logs):
        print(f'{epoch}/{self.params["epochs"]} -- loss {logs["loss"]}',
              end='\r')


def get_callbacks(monitor, mode, args):
    chp = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.path_output, "model.h5"),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        period=args.epoch_num // 100,
        verbose=1)

    tb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.path_output, "graph"),
        histogram_freq=0,
        write_graph=True,
        write_grads=False)

    pl = PrettyLogger(display=20)

    return [chp, tb, pl]
