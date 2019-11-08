from collections import namedtuple
from collections import OrderedDict


GRID_DIM = namedtuple('GRID_DIM', 'x y')(64, 512)

SOLUTION_DIMS = OrderedDict()
SOLUTION_DIMS['deformations'] = 3
SOLUTION_DIMS['stresses'] = 4
SOLUTION_DIMS['currents'] = 3
SOLUTION_DIMS['fluxes'] = 3

PHYSICAL_DIMS = OrderedDict()
PHYSICAL_DIMS['strain'] = 3
PHYSICAL_DIMS['stress'] = 4
PHYSICAL_DIMS['electric_potential'] = 1
PHYSICAL_DIMS['electric_current'] = 2
PHYSICAL_DIMS['li_concentration'] = 1
PHYSICAL_DIMS['li_flux'] = 2

PHYSICAL_DIMS_SCALAR = OrderedDict()
PHYSICAL_DIMS_SCALAR['strain_0'] = 1
PHYSICAL_DIMS_SCALAR['strain_1'] = 1
PHYSICAL_DIMS_SCALAR['strain_2'] = 1
PHYSICAL_DIMS_SCALAR['stress_0'] = 1
PHYSICAL_DIMS_SCALAR['stress_1'] = 1
PHYSICAL_DIMS_SCALAR['stress_2'] = 1
PHYSICAL_DIMS_SCALAR['stress_3'] = 1
PHYSICAL_DIMS_SCALAR['electric_potential'] = 1
PHYSICAL_DIMS_SCALAR['electric_current_0'] = 1
PHYSICAL_DIMS_SCALAR['electric_current_1'] = 1
PHYSICAL_DIMS_SCALAR['li_concentration'] = 1
PHYSICAL_DIMS_SCALAR['li_flux_0'] = 1
PHYSICAL_DIMS_SCALAR['li_flux_1'] = 1

TARGET_DIM = sum([SOLUTION_DIMS[name] for name in SOLUTION_DIMS])

FEATURE_TO_DIM = dict(
    boundary=10,
    surface=3,
    edge=10,
)

BOUNDARY_NAMES = [
    'leftEdgeSPE',
    'leftEdgeCathode',
    'leftEdgeAnode',
    'yZero',
    'rightEdgeAnode',
    'rightEdgeSPE',
    'rightEdgeCathode',
    'yTop',
    # 'intCatElCathode',
    'intCatElSPE',
    'intAnElSPE',
    # 'intAnElAnode'
]

SURFACE_NAMES = [
    'Anode',
    'Cathode',
    'SP'
]

