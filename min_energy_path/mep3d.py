from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import time

from min_energy_path.path_helpers import (get_standard_factor, get_good_path_start_samples, calculate_good_paths,
                                          generate_sphere_slice_path, breakdown_good_path)
from min_energy_path.gaussian_params import medium3d
from min_energy_path.points_sphere import create_sphere_points


start_time = time.time()

# First set up some constants we're going to need
N_SPANNING_GAP = 7
N_VARIABLES = N_SPANNING_GAP + 1

# Constants relating to the gaussian field
MIX_PARAMS = medium3d()

POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP)

# Plot the space we're exploring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_scatter_with_minima(SPHERE, MINIMA_COORDS, ax)

# quality_function = get_standard_factor(
#     points_info=POINTS_INFO,
#     mix_params=MIX_PARAMS,
#     length_cutoff=4,
#     tuning_dist=.25,
#     tuning_strength=1.5,
#     tuning_strength_diff=1,
#     tuning_grad=0.5,
#     tuning_second_order=1/47
# )

quality_function = get_standard_factor(
    points_info=POINTS_INFO,
    mix_params=MIX_PARAMS,
    length_cutoff=4,
    tuning_dist=1/20,
    tuning_strength=1/8,
    tuning_strength_diff=2,
    tuning_grad=0,
    tuning_second_order=0
)

ftree = generate_sphere_slice_path(
    quality_function, POINTS_INFO,
    n_variables=N_SPANNING_GAP+1,
    n_slices_behind=1,
    n_slices_ahead=2
)

vars = list(ftree.get_variables())
var_middle = vars[len(vars) // 2]

traversal, run = ftree.run_max_quality_forward(var_middle)
good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=4)
good_paths_info = calculate_good_paths(good_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

print(f'Running time {time.time() - start_time}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(*MIX_PARAMS['centre'], c=MIX_PARAMS['magnitude'], s=40*MIX_PARAMS['sigma'])

for i, path_info in enumerate(good_paths_info):
    ax.plot(*path_info['path'], label=f'{i}')
    print(i, path_info['value'])


plt.legend()
plt.show()

#breakdown_good_path(good_paths_info[3], ftree, quality_function, POINTS_INFO)
