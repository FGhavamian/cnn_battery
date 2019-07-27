import pandas as pd

from trainer.utils.util import *

MODEL_NAME = 'hydra_v2'
FEATURE_NAME = 'boundary_edge_surface'

DIR_TFRECORDS = os.path.join('data', 'processed', FEATURE_NAME, 'tfrecords')
DIR_MODELS = os.path.join('output', MODEL_NAME, FEATURE_NAME)


def evaluate():
    path_models = glob.glob(os.path.join(DIR_MODELS, 'cnn_*', 'model.h5'))

    # make evaluations
    for path_model in path_models:
        command = (
            'python3 -m trainer.evaluate'
            ' --path-model={}'
            ' --dir-tfrecords={}'
        ).format(path_model, DIR_TFRECORDS)

        os.system(command)

    # aggregate evaluations
    def to_score_path(x):
        return os.path.join(*x.split('/')[:-1], 'score.json')

    scores = [read_json(to_score_path(path_model)) for path_model in path_models]

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

    agg_score_path = os.path.join(*path_models[0].split('/')[:-2])

    write_json(os.path.join(agg_score_path, 'train.json'), agg_scores_train)
    write_json(os.path.join(agg_score_path, 'test.json'), agg_scores_test)


if __name__ == '__main__':
    evaluate()
