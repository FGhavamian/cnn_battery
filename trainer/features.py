from scipy.spatial.distance import cdist
from scipy.interpolate import NearestNDInterpolator
import pandas as pd

from trainer.utils.util import *
from trainer.names import *


class FeatureMaker:
    def __init__(self, mesh_data, feature_to_include):
        """
        This class makes features on the domain of the problem.
        Args:
            mesh_data ():
        """
        if isinstance(feature_to_include, list): self.feature_to_include = feature_to_include
        else: self.feature_to_include = [feature_to_include]

        self.mesh_data = mesh_data

        # submit specific feature makers here
        self.feature_makers = dict(
            surface=FeatureMakerSurface,
            boundary=FeatureMakerBoundary,
            edge=FeatureMakerEdge
        )

    def __call__(self):
        features = []
        for feature_name in self.feature_to_include:
            feature_maker = self.feature_makers[feature_name](self.mesh_data)
            feature = feature_maker()

            features.append(feature)

        df_features = pd.concat(features, axis=1)

        return df_features


class FeatureMakerEdge:
    def __init__(self, mesh_data):
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented')
        self.mesh_data = mesh_data
        self.node_coord = mesh_data['coord']

    def __call__(self):
        features = []
        # for _, boundary_node in self.mesh_data["groups_node"].items():
        for boundary_name in BOUNDARY_NAMES:
            boundary_node = self.mesh_data["groups_node"][boundary_name]

            boundary_node_encoding = np.zeros(self.node_coord.shape[0], dtype=np.float32)
            boundary_node_encoding[boundary_node] = 1.0

            f = NearestNDInterpolator(self.node_coord, boundary_node_encoding)
            surface_grid_encoding = f(self.grid_coord)

            features += [surface_grid_encoding]

        return np.stack(features, axis=1)


class FeatureMakerSurface:
    def __init__(self, grid_coord, mesh_data):
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented')
        self.grid_coord = grid_coord
        self.mesh_data = mesh_data
        self.node_coord = mesh_data['coord']

    def __call__(self):
        features = []
        # for _, surface_node in self.mesh_data['groups_element_nodes'].items():
        for surface_name in SURFACE_NAMES:
            surface_node = self.mesh_data["groups_element_nodes"][surface_name]

            surface_node_encoding = np.zeros(self.node_coord.shape[0], dtype=np.float32)
            surface_node_encoding[surface_node] = 1.0

            f = NearestNDInterpolator(self.node_coord, surface_node_encoding)
            surface_grid_encoding = f(self.grid_coord)

            features += [surface_grid_encoding]

        return np.stack(features, axis=1)


class FeatureMakerBoundary:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data
        # self.mesh_data = mesh_data
        # self.node_coord = mesh_data['coord']

    def __call__(self):
        df_features = pd.DataFrame()

        for group, some_nodes in self.mesh_data['groups_nodes'].items():
            some_coord = self.mesh_data['coord'][some_nodes]
            dist_node_to_boundary = cdist(self.mesh_data['coord'], some_coord)

            dist_node_to_boundary = np.min(dist_node_to_boundary, axis=1)
            dist_node_to_boundary = self.dist_radial(dist_node_to_boundary)

            df_features[group] = dist_node_to_boundary

        return df_features

        # for _, boundary_node in self.mesh_data["groups_node"].items():
        # for boundary_name in BOUNDARY_NAMES:
        #     boundary_node = self.mesh_data["groups_node"][boundary_name]
        #
        #     boundary_node_coord = self.get_coord_of(boundary_node)
        #
        #     boundary_node_to_grid_dist = self.closest_distance_to(boundary_node_coord)
        #
        #     boundary_feature = self.dist_radial(boundary_node_to_grid_dist)
        #
        #     features += [boundary_feature]
        #
        # return np.stack(features, axis=1)

    @staticmethod
    def dist_radial(x):
        return np.exp(-10 * x/(x.max()+1e-8))


if __name__ == '__main__':
    mesh_data = dict()
    mesh_data['name'] = 'cnn_test'
    mesh_data['coord'] = np.array(
        [[0, 0],
         [2, 0],
         [2, 2],
         [0, 2],
         [1, 1]]
    )
    mesh_data['groups_node'] = dict(
        left=[0, 3],
    )
    mesh_data['groups_element_nodes'] = dict(
        one=[0, 1, 4],
        two=[1, 2, 4]
    )

    grid_coord = np.array(
        [[0, -.5],
         [1, -.5],
         [2, -.5],
         [0, .5],
         [1, .5],
         [2, .5],
         [0, 1.5],
         [1, 1.5],
         [2, 1.5],
         [0, 2.5],
         [1, 2.5],
         [2, 2.5]]
    )

    f = FeatureMaker(grid_coord, mesh_data, 'edge')()

    print(f)
