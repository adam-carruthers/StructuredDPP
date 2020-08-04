import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from min_energy_path.path_helpers import (get_standard_factor, get_good_path_start_samples, calculate_good_paths,
                                          generate_sphere_slice_path, breakdown_good_path)
from min_energy_path.gaussian_params import medium3d
from min_energy_path.points_sphere import create_sphere_points


start_time = time.time()

# First set up some constants we're going to need
N_SPANNING_GAP = 7

# Constants relating to the gaussian field
MIX_MAG, MIX_SIG, MIX_CENTRE, MINIMA_COORDS, XBOUNDS, YBOUNDS, ZBOUNDS = medium3d()

POINTS_INFO = create_sphere_points(MINIMA_COORDS, N_SPANNING_GAP)
# TODO: Make it so it just uses a dictionary of params
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

# Plot the space we're exploring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_scatter_with_minima(SPHERE, MINIMA_COORDS, ax)

intermediate_factor_quality = get_standard_factor(
    sphere=SPHERE,
    mix_mag=MIX_MAG, mix_sig=MIX_SIG, mix_centre=MIX_CENTRE,
    point_distance=POINT_DISTANCE,
    length_cutoff=4,
    tuning_dist=.25,
    tuning_strength=1.5,
    tuning_strength_diff=1,
    tuning_grad=0.5,
    tuning_second_order=1/47
)

ftree, nodes_to_add = generate_sphere_slice_path(
    intermediate_factor_quality, ROOT_INDEX, TAIL_INDEX, ROOT_DIR_INDEX, DIR_COMPONENT, SPHERE_INDEX, SPHERE_BEFORE,
    n_variables=N_SPANNING_GAP+1,
    n_slices_behind=1,
    n_slices_ahead=2
)

index_of_middle = len(nodes_to_add) // 2
var_middle = nodes_to_add[index_of_middle + (index_of_middle % 2)]  # Needs to be an even number

traversal, run = ftree.run_max_quality_forward(var_middle)
good_path_start = get_good_path_start_samples(var_middle, run, SPHERE_BEFORE, POINT_DISTANCE, n_per_group=4)
good_path_info = calculate_good_paths(good_path_start, var_middle, traversal, run, ftree)

print(f'Running time {time.time() - start_time}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(*MIX_CENTRE, c=MIX_MAG, s=40*MIX_SIG)

for grp, good_max_idx, good_max, assignment in good_path_info:
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    ax.plot(*points, label=f'{grp}')
    print(grp, good_max)


plt.legend()
plt.show()
