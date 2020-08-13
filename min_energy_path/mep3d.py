from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import time

from min_energy_path import neb
from min_energy_path.path_helpers import (get_standard_transition_quality_function, get_good_path_start_samples,
                                          calculate_good_paths,
                                          generate_path_ftree, breakdown_good_path, generate_path_ftree_better)
import min_energy_path.gaussian_params as mix_params
from min_energy_path.points_sphere import create_sphere_points


start_time = time.time()

# First set up some constants we're going to need
N_SPANNING_GAP = 7

for param_choice in [mix_params.medium3d]:#, mix_params.medium3d]:
    # Constants relating to the gaussian field
    MIX_PARAMS = param_choice()

    POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP)

    # Plot the space we're exploring
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plot_scatter_with_minima(SPHERE, MINIMA_COORDS, ax)

    # quality_function = get_standard_transition_quality_function(
    #     points_info=POINTS_INFO,
    #     mix_params=MIX_PARAMS,
    #     length_cutoff=4,
    #     tuning_dist=0.01,
    #     tuning_strength=1,
    #     tuning_strength_diff=2,
    #     tuning_grad=0,
    #     tuning_second_order=0
    # )
    #
    # ftree = generate_path_ftree(
    #     quality_function, POINTS_INFO,
    #     n_spanning_gap=N_SPANNING_GAP,
    #     n_slices_behind=1,
    #     n_slices_ahead=2
    # )
    
    ftree = generate_path_ftree_better(
        POINTS_INFO, MIX_PARAMS,
        length_cutoff=4,
        tuning_dist=0.01,
        tuning_strength=1,
        tuning_strength_diff=2,
        n_spanning_gap=N_SPANNING_GAP,
        n_slices_behind=1,
        n_slices_ahead=2
    )


    vars = list(ftree.get_variables())
    var_middle = vars[len(vars) // 2]

    traversal, run = ftree.run_max_quality_forward(var_middle)
    good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=50)
    good_paths_info = calculate_good_paths(good_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

    print(f'Running time {time.time() - start_time}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(*MIX_PARAMS['centre'], c=MIX_PARAMS['magnitude'], s=40*MIX_PARAMS['sigma'])

    # Get the top 3 best paths
    best_paths_info = []
    for i in range(3):
        best = max(good_paths_info, key=lambda x: x['value'])
        good_paths_info.remove(best)
        best_paths_info.append(best)

    for i, path_info in enumerate(best_paths_info):
        lines = ax.plot(*path_info['path'], label=f'{i}')
        print(i, path_info['value'])
        # neb_path = neb.neb_mep(path_info, POINTS_INFO, MIX_PARAMS, n_iterations=5000)
        # plt.plot(*neb_path, c=lines[0].get_color(), dashes=(2, 2))


    plt.legend()
    plt.show()

# breakdown_good_path(best_paths_info[0], ftree, quality_function, POINTS_INFO)

# fact2 = next(iter(ftree.get_factors()))
# t_quals = fact2.transition_qualities
