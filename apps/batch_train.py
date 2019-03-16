from datetime import datetime
import os
import glob

from trainer.util import *


NUM_TESTS = 5

MODEL_NAMES = [
    # 'hydra_v0',
    # 'hydra_scalar_v0',
    # 'hydra_v1',
    # 'hydra_scalar_v1',
    # 'hydra_unet_v0',
    # 'hydra_v2',
    'simple_cnn',
    # 'hydra_scalar_v2'
]

FEATURE_NAMES = [
    'boundary_edge_surface',
    # 'boundary_surface',
    # 'boundary_edge',
    # 'surface_edge',
    # 'boundary',
    # 'surface',
    # 'edge'
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


def train(job_name, feature_name, model_name):
    feature_name = '_'.join(sorted(feature_name.split('_')))

    path_tfrecords = os.path.join('data', 'processed', feature_name, 'tfrecords')

    command = (
        'python -m trainer.train'
        ' --model-name={}'
        ' --job-name={}'
        ' --feature-name={}'
        ' --path-tfrecords={}'
    ).format(model_name, job_name, feature_name, path_tfrecords)

    os.system(command)


def main():
    for feature_name in FEATURE_NAMES:
        make_data(feature_name)

        for model_name in MODEL_NAMES:

            for num_test in range(NUM_TESTS):
                print('running ', model_name, ' test number ', num_test)
                job_name = make_job_name()

                train(job_name, feature_name, model_name)


if __name__ == '__main__':
    main()
