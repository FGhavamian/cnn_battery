import argparse

from trainer.data import *
from trainer.utils.util import *
from trainer.names import *


def load_model(path_model):
    metric_funcs = make_metrics()
    metric_names = []
    for sol_name, sol_dim in SOLUTION_DIMS.items():
        for i in range(sol_dim):
            metric_names.append(sol_name + '_' + str(i))

    metric_names.append('all')

    custom_metrics = dict(zip(metric_names, metric_funcs))

    return keras.models.load_model(
        filepath=path_model,
        custom_objects=custom_metrics)


def evaluate(dataset, model):
    score = model.evaluate(dataset, steps=1)

    return dict(zip(model.metrics_names, score))


def make_datasets(args):
    dataset_train = make_dataset(
        path_tfrecords=args.dir_tfrecords,
        batch_size=128, mode='train')

    dataset_test = make_dataset(
        path_tfrecords=args.dir_tfrecords,
        batch_size=128, mode='test')

    return dataset_train, dataset_test


def main(args):
    dataset_train, dataset_test = make_datasets(args)

    model = load_model(args.path_model)

    score_train = evaluate(dataset_train, model)
    score_test = evaluate(dataset_test, model)
    scores = dict(train=score_train, test=score_test)

    scores_path = os.path.join(*args.path_model.split('/')[:-1], 'score.json')
    write_json(scores_path, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path-model',
        help='path to the model directory',
        required=True)

    parser.add_argument(
        '--dir-tfrecords',
        help='path to tfrecords files',
        required=True
    )

    args = parser.parse_args()

    main(args)

