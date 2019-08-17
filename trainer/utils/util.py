import os
import glob
import pickle
import time
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np
from bs4 import BeautifulSoup

from trainer.names import *


def average_relative_error(name, index=None):
    def func(y_true, y_pred):
        if index:
            y_true = y_true[:, :, :, index]
            y_pred = y_pred[:, :, :, index]

        nom = keras.backend.mean(keras.backend.square(y_true - y_pred))
        denom = keras.backend.mean(keras.backend.square(y_true))

        avg_rel_err = nom / (denom + 1)

        return avg_rel_err

    func.__name__ = name
    return func


def make_metrics():
    # TODO: pass mask here to compute number of nonzero grid points
    indexes = [i for i in range(TARGET_DIM)]
    names = [[name + '_' + str(i) for i in range(size)] for name, size in SOLUTION_DIMS.items()]
    names = [name for name_list in names for name in name_list]

    ares = [average_relative_error(*index_name) for index_name in zip(names, indexes)]
    are = average_relative_error(name='all')

    metrics = ares + [are]

    return metrics


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


def parse_msh(file_path):
    with open(file_path) as file:
        content = file.readlines()

    idx_start_physical_names = 0
    idx_end_physical_names = 0
    idx_start_nodes = 0
    idx_end_nodes = 0
    idx_start_elements = 0
    idx_end_elements = 0

    for idx, line in enumerate(content):
        line = line.replace("\n", "")

        if line == "$PhysicalNames":
            idx_start_physical_names = idx + 2

        elif line == "$EndPhysicalNames":
            idx_end_physical_names = idx

        elif line == "$Nodes":
            idx_start_nodes = idx + 2

        elif line == "$EndNodes":
            idx_end_nodes = idx

        elif line == "$Elements":
            idx_start_elements = idx + 2

        elif line == "$EndElements":
            idx_end_elements = idx

    # phycical name to physical label
    physical_names_labels_string = content[idx_start_physical_names:idx_end_physical_names]
    physical_labels_names = {}
    for physical_name_label_string in physical_names_labels_string:
        physical_name_label_string = physical_name_label_string.replace("\n", "")
        physical_name_label_string = physical_name_label_string.split(" ")
        physical_labels_names.update(
            {int(physical_name_label_string[1]): physical_name_label_string[2].replace('"', "")})

    physical_names = [v for _, v in physical_labels_names.items()]

    # node number to coordinates
    nodes_coords_string = content[idx_start_nodes:idx_end_nodes]
    nodes_coords = []
    for node_coords_string in nodes_coords_string:
        node_coords_string = node_coords_string.replace("\n", "")
        node_coords_string = node_coords_string.split(" ")
        node_coords_string = node_coords_string[1:-1]
        node_coords = [float(n) for n in node_coords_string]
        nodes_coords.append(node_coords)

    # element groups
    elements_nodes_string = content[idx_start_elements:idx_end_elements]
    groups_elements = {v: [] for v in physical_names}
    for i, element_nodes_string in enumerate(elements_nodes_string):
        element_nodes_string = element_nodes_string.replace("\n", "")
        element_nodes_string = element_nodes_string.split(" ")
        groups_elements[physical_labels_names[int(element_nodes_string[3])]] += [i]

    # element number to element nodes
    elements_nodes_string = content[idx_start_elements:idx_end_elements]
    elements_nodes = []
    for i, element_nodes_string in enumerate(elements_nodes_string):
        element_nodes_string = element_nodes_string.replace("\n", "")
        element_nodes_string = element_nodes_string.split(" ")
        elements_nodes.append([int(e) - 1 for e in element_nodes_string[5:]])

    # groups to nodes
    groups_nodes = {v: [] for v in physical_names}
    for physical_name in physical_names:
        elements = groups_elements[physical_name]
        nodes = [n for e in elements for n in elements_nodes[e]]
        nodes = list(set(nodes))
        groups_nodes[physical_name] = nodes

    return nodes_coords, groups_nodes, [p for p in physical_names if p != "surface"]


class MeshParser:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            mesh = file.read()

        self.soup = BeautifulSoup(mesh, 'lxml')

    def get_node_coord(self):
        node_coord = self.soup.find('nodes').text
        node_coord = node_coord.replace(';', '')
        node_coord = node_coord.split('\n')
        node_coord = [n for n in node_coord if n]
        node_coord = [[float(node) for node in nodes.split(' ')] for nodes in node_coord]
        node_coord = [[n for n in nodes[1:]] for nodes in node_coord]

        return node_coord

    def get_groups_node(self):
        groups_node_soup = self.soup.find_all('nodegroup')

        groups_node = {}
        for group_node_soup in groups_node_soup:
            group_name = group_node_soup.attrs['name']

            nodes = group_node_soup.text
            nodes = nodes.replace('}', '')
            nodes = nodes.replace('{', '')
            nodes = nodes.replace(',', '')
            nodes = nodes.split(' ')
            nodes = [int(n) for n in nodes]

            groups_node.update({group_name: nodes})

        # remove the 999 group
        groups_node.pop('999')

        return groups_node

    def get_element_connect(self):
        element_nodes = self.soup.find('elements').text
        element_nodes = element_nodes.replace(';', '')
        element_nodes = element_nodes.split('\n')
        element_nodes = [n for n in element_nodes if n]
        element_nodes = [[int(node) for node in nodes.split(' ')] for nodes in element_nodes]
        element_nodes = {nodes[0]: [n for n in nodes[1:]] for nodes in element_nodes}
        # element_nodes = [[n for n in nodes[1:]] for nodes in element_nodes]
        # element_nodes = np.array(element_nodes)


        return element_nodes

    def get_groups_element(self):
        groups_element_soup = self.soup.find_all('elementgroup')

        groups_element = {}
        for group_element_soup in groups_element_soup:
            group_name = group_element_soup.attrs['name']

            elements = group_element_soup.text
            elements = elements.replace('}', '')
            elements = elements.replace('{', '')
            elements = elements.replace(',', '')
            elements = elements.split(' ')
            elements = [int(e) for e in elements]

            groups_element.update({group_name: elements})

        return groups_element

    def get_nodes_in_element_groups(self):
        groups_element = self.get_groups_element()
        element_connect = self.get_element_connect()

        groups_element_nodes = {group_name: [] for group_name in groups_element}

        for group_name, elements in groups_element.items():
            for element in elements:
                nodes = element_connect[element]
                groups_element_nodes[group_name] += nodes

        return groups_element_nodes


def read_mesh(file_path):
    if file_path.split(".")[-1] == "msh":
        return parse_msh(file_path)

    elif file_path.split(".")[-1] == "mesh":
        mesh_parse = MeshParser(file_path)

        nodes_coord = mesh_parse.get_node_coord()
        groups_node = mesh_parse.get_groups_node()
        groups_element_nodes = mesh_parse.get_nodes_in_element_groups()
        element_connect = mesh_parse.get_element_connect()

        element_connect = np.array([element_connect[e] for e in sorted(list(element_connect.keys()))])

        return np.array(nodes_coord, np.float32), groups_node, groups_element_nodes, element_connect


def read_vtu(file_path):
    with open(file_path) as file:
        content = file.readlines()

    countingNodes = False
    countingDeformations = False
    countingStresses = False
    countingCurrent = False
    countingFluxes = False
    for idx, line in enumerate(content):
        line = line.replace("\n", "")

        if "coordinates" in line:
            idx_start_nodes = idx + 1
            countingNodes = True

        elif "/DataArray" in line and countingNodes is True:
            idx_end_nodes = idx
            countingNodes = False

        elif "connectivity" in line:
            idx_start_connect = idx + 1
            countingConnect = True

        elif "/DataArray" in line and countingConnect is True:
            idx_end_connect = idx
            countingConnect = False

        elif "deformations" in line:
            idx_start_deformations = idx + 1
            countingDeformations = True

        elif "/DataArray" in line and countingDeformations is True:
            idx_end_deformations = idx
            countingDeformations = False

        elif "stresses" in line:
            idx_start_stresses = idx + 1
            countingStresses = True

        elif "/DataArray" in line and countingStresses is True:
            idx_end_stresses = idx
            countingStresses = False

        elif "current" in line:
            idx_start_current = idx + 1
            countingCurrent = True

        elif "/DataArray" in line and countingCurrent is True:
            idx_end_current = idx
            countingCurrent = False

        elif "fluxes" in line:
            idx_start_fluxes = idx + 1
            countingFluxes = True

        elif "/DataArray" in line and countingFluxes is True:
            idx_end_fluxes = idx
            countingFluxes = False

    # nodes to coords
    nodes_coords_string = content[idx_start_nodes:idx_end_nodes]
    nodes_coords = []
    for node_coords_string in nodes_coords_string:
        node_coords_string = node_coords_string.replace("\n", "") \
                                 .replace("\t", "", 5) \
                                 .replace("\t", " ", 2) \
                                 .replace("\t", "", 1) \
                                 .split(" ")[:-1]
        node_coords = [float(n) for n in node_coords_string]
        nodes_coords.append(node_coords)

    # element to nodes
    connects_string = content[idx_start_connect:idx_end_connect]
    connects = []
    for connect_string in connects_string:
        connect_string = connect_string.replace("\n", "") \
                                 .replace("\t", "", 5) \
                                 .replace("\t", " ", 2) \
                                 .replace("\t", "", 1) \
                                 .split(" ")
        connect = [int(n) for n in connect_string]
        connects.append(connect)

    # nodal deformations
    deformations_string = content[idx_start_deformations:idx_end_deformations]
    deformations = []
    for deformation_string in deformations_string:
        deformation_string = deformation_string.replace("\n", "") \
            .replace("\t", "", 5) \
            .replace("\t", " ", 2) \
            .replace("\t", "", 1) \
            .split(" ")

        deformation = [float(n) for n in deformation_string]
        deformations.append(deformation)

    # nodal stresses
    stresses_string = content[idx_start_stresses:idx_end_stresses]
    stresses = []
    for stress_string in stresses_string:
        stress_string = stress_string.replace("\n", "") \
            .replace("\t", "", 5) \
            .replace("\t", " ", 5) \
            .replace("\t", "", 1) \
            .split(" ")

        stress = [float(n) for n in stress_string]
        stresses.append(stress)

    # nodal current
    currents_string = content[idx_start_current:idx_end_current]
    currents = []
    for current_string in currents_string:
        current_string = current_string.replace("\n", "") \
            .replace("\t", "", 5) \
            .replace("\t", " ", 3) \
            .replace("\t", "", 1) \
            .split(" ")

        current = [float(n) for n in current_string]
        currents.append(current)

    # nodal fluxes
    fluxes_string = content[idx_start_fluxes:idx_end_fluxes]
    fluxes = []
    for flux_string in fluxes_string:
        flux_string = flux_string.replace("\n", "") \
            .replace("\t", "", 5) \
            .replace("\t", " ", 3) \
            .replace("\t", "", 1) \
            .split(" ")

        flux = [float(n) for n in flux_string]
        fluxes.append(flux)

    # convert to numpy
    nodes_coords = np.array(nodes_coords, dtype=np.float32)
    deformations = np.array(deformations, dtype=np.float32)
    connects = np.array(connects, dtype=np.int32)
    stresses = np.array(stresses, dtype=np.float32)
    currents = np.array(currents, dtype=np.float32)
    fluxes = np.array(fluxes, dtype=np.float32)

    # !!! first flux is the concentration, c0 = 1.5e-15
    fluxes[:, 0] /= 1.5e-15

    # remove post processed fields
    stresses = stresses[:, :4]
    currents = currents[:, :-1]
    fluxes = fluxes[:, :-1]

    sol = dict(deformations=deformations, stresses=stresses, currents=currents, fluxes=fluxes)

    return nodes_coords, connects, sol


class VtuWriter:
    def __init__(self, report_dir, file_name, coord, connect, pred):
        self.report_dir = report_dir
        self.file_name = file_name
        self.num_points = coord.shape[0]
        self.num_cells = len(connect)
        self.coord = coord
        self.connect = connect

        self.fields = dict(
            deformations=pred[:, :3],
            stresses=pred[:, 3:7],
            current=pred[:, 7:10],
            fluxes=pred[:, 10:]
        )

    def __call__(self):
        empty_vtu_str = self.__make_wrapper()
        vtu_str = self.__fill_wrapper(empty_vtu_str)

        self.__write_to_file(vtu_str)

    def __make_wrapper(self):
        return ('<VTKFile type="UnstructuredGrid" version="0.1">\n'
                '\t<UnstructuredGrid>\n'
                '\t\t<Piece NumberOfPoints="{num_points}" NumberOfCells="{num_cells}">\n'
                '\t\t\t<Points>\n'
                
                '\t\t\t\t<DataArray type="Float64" Name="coordinates" NumberOfComponents="3"  format="ascii" >\n'
                '{coord}\n'
                '\t\t\t\t</DataArray>\n'
                
                '\t\t\t</Points>\n'
                '\t\t\t<Cells>\n'
                
                '\t\t\t\t<DataArray  type="Int64"  Name="connectivity"  format="ascii">\n'
                '{connect}\n'
                '\t\t\t\t</DataArray>\n'

                '\t\t\t\t<DataArray  type="Int32"  Name="offsets"  format="ascii">\n'
                '{offsets}\n'
                '\t\t\t\t</DataArray>\n'

                '\t\t\t\t<DataArray  type="UInt8"  Name="types"  format="ascii">\n'
                '{types}\n'
                '\t\t\t\t</DataArray>\n'

                '\t\t\t</Cells>\n'
                '\t\t\t<PointData  Scalars="Data">\n'

                '\t\t\t\t<DataArray  type="Float64"  Name="deformations" NumberOfComponents="3" format="ascii">\n'
                '{deformations}\n'
                '\t\t\t\t</DataArray>\n'
                
                '\t\t\t\t<DataArray  type="Float64"  Name="stresses" NumberOfComponents="4" format="ascii">\n'
                '{stresses}\n'
                '\t\t\t\t</DataArray>\n'
                
                '\t\t\t\t<DataArray  type="Float64"  Name="current" NumberOfComponents="3" format="ascii">\n'
                '{current}\n'
                '\t\t\t\t</DataArray>\n'
                
                '\t\t\t\t<DataArray  type="Float64"  Name="fluxes" NumberOfComponents="3" format="ascii">\n'
                '{fluxes}\n'
                '\t\t\t\t</DataArray>\n'

                '\t\t\t</PointData>\n'
                '\t\t</Piece>\n'
                '\t</UnstructuredGrid>\n'
                '</VTKFile>'
                )

    def __fill_wrapper(self, wrapper):
        return wrapper.format(
            **dict(
                num_points=self.num_points,
                num_cells=self.num_cells,
                coord=self.__coord_str(),
                connect=self.__connect_str(),
                offsets=self.__offsets_str(),
                types=self.__types_str(),
                deformations=self.__deformations_str(),
                stresses=self.__stresses_str(),
                current=self.__current_str(),
                fluxes=self.__fluxes_str()
            )
        )

    def __coord_str(self):
        string = ''
        tab_str = '\t'*5
        for n in self.coord:
            string += tab_str + '{:0.4f} {:0.4f} 0.0\n'.format(*n)

        return string

    def __connect_str(self):
        string = ''
        tab_str = '\t' * 5
        for _, n in self.connect.items():
            string += tab_str + '{} {} {}\n'.format(*n)

        return string

    def __offsets_str(self):
        string = ''
        tab_str = '\t' * 5
        for c in range(self.num_cells):
            string += tab_str + '{}\n'.format((c+1)*3)

        return string

    def __types_str(self):
        string = ''
        tab_str = '\t' * 5
        for c in range(self.num_cells):
            string += tab_str + '5\n'

        return string

    def __deformations_str(self):
        string = ''
        tab_str = '\t' * 5
        for n in self.fields['deformations']:
            string += tab_str + '{} {} {}\n'.format(*n)

        return string

    def __stresses_str(self):
        string = ''
        tab_str = '\t' * 5
        for n in self.fields['stresses']:
            string += tab_str + '{} {} {} {}\n'.format(*n)

        return string

    def __current_str(self):
        string = ''
        tab_str = '\t' * 5
        for n in self.fields['current']:
            string += tab_str + '{} {} {}\n'.format(*n)

        return string

    def __fluxes_str(self):
        string = ''
        tab_str = '\t' * 5
        for n in self.fields['fluxes']:
            string += tab_str + '{} {} {}\n'.format(*n)

        return string

    def __write_to_file(self, vtu_str):
        with open(os.path.join(self.report_dir, self.file_name + '.vtu'), 'w') as file:
            file.write(vtu_str)


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

    job_path = job_paths[job_path_idx]

    return job_path


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

