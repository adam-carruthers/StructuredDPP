import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from min_energy_path.gaussian_field import plot_gaussian
import min_energy_path.gaussian_params as mix_params
from min_energy_path.points_sphere import create_sphere_points
from min_energy_path.path_helpers import (get_standard_transition_quality_function, get_good_path_start_samples, calculate_good_paths,
                                          breakdown_good_path, generate_path_ftree_better)
from min_energy_path import neb

start_time = time.time()

# First set up some constants we're going to need
N_SPANNING_GAP = 10
N_VARIABLES = N_SPANNING_GAP + 2

for param_choice in [mix_params.hashtag_blocked]:  # , mix_params.starter, mix_params.complex2d, mix_params.hashtag_blocked, mix_params.even_simplerer]:
    # Constants relating to the gaussian field
    MIX_PARAMS = param_choice()

    POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP)

    ftree = generate_path_ftree_better(
        POINTS_INFO, MIX_PARAMS,
        length_cutoff=4,
        tuning_dist=0.02,
        tuning_strength=1,
        tuning_strength_diff=1.5,
        n_spanning_gap=N_SPANNING_GAP,
        n_slices_behind=1,
        n_slices_ahead=2
    )

    vars = list(ftree.get_variables())
    var_middle = vars[(len(vars) // 2)-1]

    traversal, run = ftree.run_max_quality_forward(var_middle)
    good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=4)
    good_paths_info = calculate_good_paths(good_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

    print(f'Running time {time.time() - start_time}')

    # Get the top 4 best paths
    # best_paths_info = []
    # for i in range(3):
    #     best = max(good_paths_info, key=lambda x: x['value'])
    #     good_paths_info.remove(best)
    #     best_paths_info.append(best)

    plot_gaussian(MIX_PARAMS)

    for i, path_info in enumerate(good_paths_info):
        print(i, path_info['value'])
        lines = plt.plot(*path_info['path'], label=f'q{i}={path_info["value"]}')
        # neb_path_1 = neb.neb_mep(path_info, POINTS_INFO, MIX_PARAMS, k=2., n_iterations=10000)
        # plt.plot(*neb_path_1, c=lines[0].get_color(), dashes=(2, 2))

    # Get the NEB smoothed best path

    plt.legend()
    plt.show()

# breakdown_good_path(good_paths_info[3], ftree, quality_function, POINTS_INFO)
