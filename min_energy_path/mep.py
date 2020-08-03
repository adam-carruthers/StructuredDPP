import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from structured_dpp.factor_tree import *

from min_energy_path.gaussian_field import plot_gaussian
from min_energy_path.guassian_params import starter
from min_energy_path.points_sphere import create_sphere_points, plot_scatter_with_minima
from min_energy_path.path_helpers import (get_standard_factor, get_good_path_start_samples, calculate_good_paths,
                                          breakdown_good_path, generate_sphere_slice_path)

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
ROOT_INDEX = POINTS_INFO['root_index']
TAIL_INDEX = POINTS_INFO['tail_index']
ROOT_DIR_INDEX = POINTS_INFO['root_dir_index']

# # Plot the space we're exploring
# plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)
# plot_scatter_with_minima(SPHERE, MINIMA_COORDS, plt)


intermediate_factor_quality = get_standard_factor(
    sphere=SPHERE,
    mix_mag=MIX_MAG, mix_sig=MIX_SIG, mix_centre=MIX_CENTRE,
    point_distance=POINT_DISTANCE,
    length_cutoff=4,
    tuning_dist=.25,
    tuning_strength=.75,
    tuning_strength_diff=1,
    tuning_grad=0.5,
    tuning_second_order=1/30
)

ftree, nodes_to_add = generate_sphere_slice_path(
    intermediate_factor_quality, ROOT_INDEX, TAIL_INDEX, ROOT_DIR_INDEX, DIR_COMPONENT, SPHERE_INDEX, SPHERE_BEFORE,
    n_variables=N_SPANNING_GAP+1,
    n_slices_behind=1,
    n_slices_ahead=2
)

index_of_middle = len(nodes_to_add) // 2
var5 = nodes_to_add[index_of_middle - (index_of_middle % 2)]

traversal, run = ftree.run_max_quality_forward(var5)
good_start_samples = get_good_path_start_samples(var5, run, SPHERE_BEFORE, POINT_DISTANCE, 3)
good_path_info = calculate_good_paths(good_start_samples, var5, traversal, run, ftree)

print(f'Running time {time.time() - start_time}')

plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)

for i, (grp, start_val, path_value, assignment) in enumerate(good_path_info):
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    plt.plot(*points, label=f'q{i}={path_value:.3e}')

plt.legend()
plt.show()
