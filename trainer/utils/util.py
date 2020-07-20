import os
import glob
import pickle
import time
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

from trainer.names import *


def average_relative_error(name, index=None):
  def func(y_true, y_pred):
    if index:
      y_true = y_true[:, :, :, index]
      y_pred = y_pred[:, :, :, index]

    nom = keras.backend.mean(keras.backend.square(y_true - y_pred))
    denom = keras.backend.mean(keras.backend.square(y_true))

    return nom / (denom + 1)

  func.__name__ = name
  return func


def make_metrics():
  # TODO: pass mask here to compute number of nonzero grid points
  indexes = [i for i in range(TARGET_DIM)]
  names = [[name + '_' + str(i) for i in range(size)] for name, size in SOLUTION_DIMS.items()]
  names = [name for name_list in names for name in name_list]

  ares = [average_relative_error(*index_name) for index_name in zip(names, indexes)]
  are = average_relative_error(name='all')

  return ares + [are]


def r2(y_true, y_pred):
  res_square = keras.backend.square(y_true - y_pred)
  dev_square = keras.backend.square(y_true - keras.backend.mean(y_true))

  res = keras.backend.sum(res_square)
  tot = keras.backend.sum(dev_square)

  return 1 - res / (tot + keras.backend.epsilon())


def print_progress(count, total):
  print(str(count) + "/" + str(total), end="\r")


def wrap_int64(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrap_bytes(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_bytes_list(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_case_name(case):
  return case.split("/")[2].replace("_mine", "")


def chunks(l, n):
  if isinstance(l, np.ndarray):
    length = l.shape[0]
  elif isinstance(l, list):
    length = len(l)
  else:
    raise Exception(ValueError)

  for i in range(0, length, n):
    yield l[i:i + n]


def bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def numpy_to_bytes(array):
  return array.tostring()


def get_job_path():
  job_paths = glob.glob(os.path.join("output", "cnn_*"))

  # sort the list of jobs based on date and time
  job_paths.sort(key=lambda x: (x.split("_")[-2], x.split("_")[-1]))

  if len(job_paths) == 0:
      raise ValueError('No job found at ' + ' '.join(job_paths))
  else:
      for i, job_path in enumerate(job_paths):
          print('[' + str(i) + ']', " : ", job_path.split("/")[-1])

  job_path_idx = int(input("Enter job index? "))

  return job_paths[job_path_idx]


def call_and_time_counter(func, *args):
  start_time = time.time()
  out = func(*args)
  end_time = time.time()
  exec_time = end_time - start_time
  print("inefernce time: " + str(exec_time))
  return out


def save_to_pickle(output_path, obj):
  with open(output_path, 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(path):
  with open(path, 'rb') as f:
    return pickle.load(f)


def to_float32(x):
  return x.astype(np.float32)


def read_json(file_path):
  with open(file_path) as file:
    return json.load(file)


def write_json(file_path, data):
  # all float values should be that of float64 so that json can serialize it
  if isinstance(data[list(data.keys())[0]], dict):
    for k in data:
      for k1 in data[k]:
        data[k][k1] = np.float64(data[k][k1])

  with open(file_path, 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

