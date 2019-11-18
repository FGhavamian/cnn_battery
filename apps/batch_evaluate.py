import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import glob

import pandas as pd

from trainer.utils.util import read_json, write_json

EXAMPLE_NAME = 'depth_size'

DIR_EXAMPLES = os.path.join('output', EXAMPLE_NAME)


def make_scores(path_cases):
    for path_case in path_cases:
        path_models = glob.glob(os.path.join(path_case, 'cnn_*', 'model.h5'))
        print(path_models)
        for path_model in path_models:
            feature_name = path_model.split(os.path.sep)[-5]
            train_data_pct = path_model.split(os.path.sep)[2]

            dir_tfrecords = os.path.join(
                'data', 'processed', feature_name + '_' + train_data_pct, 'tfrecords')

            command = (
                'python -m trainer.evaluate'
                ' --path-model={}'
                ' --dir-tfrecords={}'
            ).format(path_model, dir_tfrecords)

            os.system(command)


def aggregate_scores(path_cases):
    for path_case in path_cases:
        path_scores = glob.glob(os.path.join(path_case, 'cnn_*', 'score.json'))
        scores = [read_json(path_score) for path_score in path_scores]

        metric_names = list(scores[0]['train'].keys())

        scores_train = {metric_name: [] for metric_name in metric_names}
        scores_test = {metric_name: [] for metric_name in metric_names}

        for score in scores:
            for metric_name in metric_names:
                scores_train[metric_name].append(score['train'][metric_name])
                scores_test[metric_name].append(score['test'][metric_name])

        df_scores_train = pd.DataFrame.from_dict(scores_train)
        df_scores_test = pd.DataFrame.from_dict(scores_test)

        agg_scores_train = df_scores_train.describe().to_dict()
        agg_scores_test = df_scores_test.describe().to_dict()

        write_json(os.path.join(path_case, 'train.json'), agg_scores_train)
        write_json(os.path.join(path_case, 'test.json'), agg_scores_test)


def evaluate():
    path_cases = glob.glob(os.path.join(DIR_EXAMPLES, '**', 'kernel_*'), recursive=True)
    make_scores(path_cases)
    aggregate_scores(path_cases)



    # # aggregate evaluations
    # def to_score_path(x):
    #     return os.path.join(*x.split('/')[:-1], 'score.json')
    #
    # scores = [read_json(to_score_path(path_model)) for path_model in path_models]
    # print(path_models)
    # metric_names = list(scores[0]['train'].keys())
    #
    # scores_train = {metric_name: [] for metric_name in metric_names}
    # scores_test = {metric_name: [] for metric_name in metric_names}
    #
    # for score in scores:
    #     for metric_name in metric_names:
    #         scores_train[metric_name].append(score['train'][metric_name])
    #         scores_test[metric_name].append(score['test'][metric_name])
    #
    # df_scores_train = pd.DataFrame.from_dict(scores_train)
    # df_scores_test = pd.DataFrame.from_dict(scores_test)
    #
    # agg_scores_train = df_scores_train.describe().to_dict()
    # agg_scores_test = df_scores_test.describe().to_dict()
    #
    # agg_score_path = os.path.join(*path_models[0].split('/')[:-2])
    #
    # write_json(os.path.join(agg_score_path, 'train.json'), agg_scores_train)
    # write_json(os.path.join(agg_score_path, 'test.json'), agg_scores_test)


    # # path_models = glob.glob(os.path.join(DIR_MODELS, 'cnn_*', 'model.h5'))
    # path_examples = glob.glob(DIR_EXAMPLES)
    #
    # # make evaluations
    # for path_example in path_examples:
    #     print(f'[INFO] evaluating example at {path_example}')
    #     path_models = glob.glob(os.path.join(path_example, '*', '*', 'cnn_*', 'model.h5'))
    #     for path_model in path_models:
    #         feature_name = path_model.split('/')[-3]
    #         dir_tfrecords = os.path.join('data', 'processed', feature_name, 'tfrecords')
    #         command = (
    #             'python3 -m trainer.evaluate'
    #             ' --path-model={}'
    #             ' --dir-tfrecords={}'
    #         ).format(path_model, dir_tfrecords)
    #
    #         # os.system(command)
    #
    #     # aggregate evaluations
    #     def to_score_path(x):
    #         return os.path.join(*x.split('/')[:-1], 'score.json')
    #
    #     scores = [read_json(to_score_path(path_model)) for path_model in path_models]
    #     print(path_models)
    #     metric_names = list(scores[0]['train'].keys())
    #
    #     scores_train = {metric_name: [] for metric_name in metric_names}
    #     scores_test = {metric_name: [] for metric_name in metric_names}
    #
    #     for score in scores:
    #         for metric_name in metric_names:
    #             scores_train[metric_name].append(score['train'][metric_name])
    #             scores_test[metric_name].append(score['test'][metric_name])
    #
    #     df_scores_train = pd.DataFrame.from_dict(scores_train)
    #     df_scores_test = pd.DataFrame.from_dict(scores_test)
    #
    #     agg_scores_train = df_scores_train.describe().to_dict()
    #     agg_scores_test = df_scores_test.describe().to_dict()
    #
    #     agg_score_path = os.path.join(*path_models[0].split('/')[:-2])
    #
    #     write_json(os.path.join(agg_score_path, 'train.json'), agg_scores_train)
    #     write_json(os.path.join(agg_score_path, 'test.json'), agg_scores_test)


if __name__ == '__main__':
    evaluate()
