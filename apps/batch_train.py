from datetime import datetime
import os

from trainer.utils.util import *

EXAMPLE_NAME = 'feature_selection'

NUM_TESTS = 5
EPOCH_NUM = 500

MODEL_NAMES = [
    'hydra_v0',
    # 'hydra_v01',
    # 'hydra_scalar_v0',
    # 'hydra_v1',
    # 'hydra_scalar_v1',
    # 'hydra_unet_v0',
    # 'hydra_v2',
    # 'simple_cnn',
    # 'hydra_scalar_v2'
]

FEATURE_NAMES = [
    'boundary_edge_surface',
    'boundary_surface',
    'boundary_edge',
    'surface_edge',
    'boundary',
    'surface',
    'edge'
]

LEARNING_RATES = [
    # 1e-2,
    1e-3,
    # 1e-4
]


def make_job_name():
    return "cnn_" + datetime.now().strftime('%Y%m%d_%H%M%S')


def make_data(features):
    if not isinstance(features, list):
        features = [features]

    command = (
        'python -m trainer.preprocess' 
        ' --features={}'
    ).format(','.join(features))

    os.system(command)


def train(job_name, feature_name, model_name, learning_rate):
    feature_name = '_'.join(sorted(feature_name.split('_')))

    path_tfrecords = os.path.join('data', 'processed', feature_name, 'tfrecords')
    ex_path = os.path.join('output', EXAMPLE_NAME, str(learning_rate), model_name, feature_name)

    command = (
        'python -m trainer.train'
        ' --model-name={}'
        ' --job-name={}'
        ' --feature-name={}'
        ' --path-tfrecords={}'
        ' --learning-rate={}'
        ' --ex-path={}'
        ' --epoch-num={}'
    ).format(model_name, job_name, feature_name, path_tfrecords, learning_rate, ex_path, EPOCH_NUM)

    os.system(command)


def main():
    for feature_name in FEATURE_NAMES:
        make_data(feature_name)
        for model_name in MODEL_NAMES:
            for learning_rate in LEARNING_RATES:
                for num_test in range(NUM_TESTS):
                    print('running ', model_name, ' test number ', num_test)
                    job_name = make_job_name()
                    train(job_name, feature_name, model_name, learning_rate)


if __name__ == '__main__':
    main()
