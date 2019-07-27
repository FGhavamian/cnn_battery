from tensorflow.python.keras.callbacks import Callback


class PrettyLogger(Callback):
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