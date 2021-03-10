import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from trainer.predict import Predictor
from trainer.utils.util import *
from trainer.names import GRID_DIM


HEAD_TYPE = 'scalar'

PATH_TFRECORDS = os.path.join(
    'data',
    'processed',
    'boundary_edge_surface_1.0',
    'tfrecords'
)

PATH_SAVED_MODELS = os.path.join(
    'output', 
    'data_percentage', 
    f'{HEAD_TYPE}', 
    '1.0',
    '0.001',
    'simple',
    'boundary_edge_surface',
    'filter_32_64',
    'kernel_7'
)


model_paths = glob.glob(
    os.path.join(
        PATH_SAVED_MODELS,
        '*',
        'model.h5'
    )
)


for model_path in model_paths:

    command = (
        'python -m trainer.evaluate'
        f' --path-model={model_path}'
        f' --head-type={HEAD_TYPE}'
        f' --dir-tfrecords={PATH_TFRECORDS}'
    )
    
    os.system(command)




