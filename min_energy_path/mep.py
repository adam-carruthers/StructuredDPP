import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from min_energy_path.path_helpers import get_standard_factor
from structured_dpp.factor_tree import *
from min_energy_path.gaussian_field import gaussian_field_for_quality, plot_gaussian
from min_energy_path.guassian_params import starter
from min_energy_path.points_sphere import create_sphere_points, plot_scatter_with_minima


start_time = time.time()


# First set up some constants we're going to need
N_SPANNING_GAP = 10
N_VARIABLES = N_SPANNING_GAP + 2

# Constants relating to the gaussian field
MIX_MAG, MIX_SIG, MIX_CENTRE, MINIMA_COORDS, XBOUNDS, YBOUNDS = starter()

POINTS_INFO = create_sphere_points(MINIMA_COORDS, N_SPANNING_GAP)
SPHERE_BEFORE = POINTS_INFO['sphere_before']
SPHERE = POINTS_INFO['sphere']
SPHERE_INDEX = np.arange(SPHERE.shape[1])
N_TOTAL = POINTS_INFO['n_total']
BASIS = POINTS_INFO['basis']
MINIMA_DISTANCE = POINTS_INFO['minima_distance']
POINT_DISTANCE = POINTS_INFO['point_distance']
N_OVERFLOW = POINTS_INFO['n_overflow']
DIR_COMPONENT = POINTS_INFO['dir_component']

# Plot the space we're exploring
plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)
plot_scatter_with_minima(SPHERE, MINIMA_COORDS, plt)

# Find the minima indices
ROOT_INDEX, TAIL_INDEX = None, None
for i, point in enumerate(SPHERE.T):
    if np.allclose(point, MINIMA_COORDS[:, 0]):
        ROOT_INDEX = i
    if np.allclose(point, MINIMA_COORDS[:, 1]):
        TAIL_INDEX = i
if None in [ROOT_INDEX, TAIL_INDEX]:
    raise ValueError("Couldn't find root or tail index.")
ROOT_DIR_INDEX = np.where(DIR_COMPONENT == SPHERE_BEFORE[0, ROOT_INDEX])[0][0]

TUNING_DIST = .25
TUNING_STRENGTH = 1.5
TUNING_STRENGTH_DIFF = 1
TUNING_GRAD = 1
TUNING_SECOND_ORDER = 1
LENGTH_CUTOFF = 4


intermediate_factor_quality = get_standard_factor(
    sphere=SPHERE,
    mix_mag=MIX_MAG, mix_sig=MIX_SIG, mix_centre=MIX_CENTRE,
    point_distance=POINT_DISTANCE,
    length_cutoff=4,
    tuning_dist=.25,
    tuning_strength=1.5,
    tuning_strength_diff=1,
    tuning_grad=0.5,
    tuning_second_order=1
)


current_var = Variable((ROOT_INDEX,), name='RootVar0')
nodes_to_add = [current_var]
for i in range(1, N_VARIABLES):
    # Add transition factor
    transition_factor = Factor(intermediate_factor_quality,
                               parent=current_var,
                               name=f'Fac{i-1}-{i}')
    nodes_to_add.append(transition_factor)

    if i == N_VARIABLES - 1:
        current_var = Variable((TAIL_INDEX,),
                               parent=transition_factor,
                               name=f'TailVar{i}')
    else:
        # Sphere slice bounds
        slice_of_dir = DIR_COMPONENT[max(ROOT_DIR_INDEX+i-2, 0):ROOT_DIR_INDEX+i+2]
        in_slice = (np.min(slice_of_dir) <= SPHERE_BEFORE[0, :]) & (SPHERE_BEFORE[0, :] <= np.max(slice_of_dir))

        current_var = Variable(SPHERE_INDEX[in_slice].T,
                               parent=transition_factor,
                               name=f'Var{i}')
    nodes_to_add.append(current_var)

ftree = FactorTree.create_from_connected_nodes(nodes_to_add)

index_of_middle = len(nodes_to_add) // 2
var5 = nodes_to_add[index_of_middle - (index_of_middle % 2)]

traversal, run = ftree.run_max_quality_forward(var5)
good_max_samples = get_good_max_samples(var5, run, 3)
assignments = [ftree.get_max_from_start_assignment(var5, good_max_idx, traversal, run)
               for good_max_idx, good_max in good_max_samples.values()]

print(f'Running time {time.time() - start_time}')

plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)

for assignment, (good_max_idx, good_max) in zip(assignments, good_max_samples.values()):
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    plt.plot(*points, label=f'q={good_max:.3e}')

plt.legend()
plt.show()
