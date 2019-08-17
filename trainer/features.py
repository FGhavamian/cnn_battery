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
        self.mesh_data = mesh_data

    def __call__(self):
        df_features = pd.DataFrame()

        for boundary_name, boundary_nodes in self.mesh_data['groups_nodes'].items():
            boundary_node_encoding = np.zeros(self.mesh_data['coord'].shape[0], dtype=np.float32)
            boundary_node_encoding[boundary_nodes] = 1.0

            df_features[boundary_name] = boundary_node_encoding

        return df_features


class FeatureMakerSurface:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data

    def __call__(self):
        df_features = pd.DataFrame()

        for surface_name, surface_nodes in self.mesh_data['groups_element_nodes'].items():
            surface_node_encoding = np.zeros(self.mesh_data['coord'].shape[0], dtype=np.float32)
            surface_node_encoding[surface_nodes] = 1.0

            df_features[surface_name] = surface_node_encoding

        return df_features


class FeatureMakerBoundary:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data

    def __call__(self):
        df_features = pd.DataFrame()

        for group, boundary_nodes in self.mesh_data['groups_nodes'].items():
            some_coord = self.mesh_data['coord'][boundary_nodes]
            dist_node_to_boundary = cdist(self.mesh_data['coord'], some_coord)

            dist_node_to_boundary = np.min(dist_node_to_boundary, axis=1)
            dist_node_to_boundary = self.dist_radial(dist_node_to_boundary)

            df_features[group] = dist_node_to_boundary

        return df_features

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

    f = FeatureMaker(mesh_data, 'surface')()

    print(f)
