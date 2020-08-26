import numpy as np
from matplotlib import pyplot as plt
import time
import logging

from min_energy_path.gaussian_field import plot_gaussian, gaussian_field
import min_energy_path.gaussian_params as gauss_params
from min_energy_path.points_sphere import create_sphere_points
from min_energy_path.path_helpers import (get_standard_transition_quality_function, get_good_path_start_samples, calculate_good_paths,
                                          breakdown_good_path, generate_path_ftree_better)
from min_energy_path import neb

logging.basicConfig(level=logging.INFO)

results = []

dims = [6]
for dim, n_iters in zip(dims, [50]):
    n_centres = 1250

    mrf_eps = []
    neb_eps = []

    for i in range(n_iters):
        start_time = time.time()

        # First set up some constants we're going to need
        N_SPANNING_GAP = 5

        # Constants relating to the gaussian field
        MIX_PARAMS = gauss_params.randomly_generated(dim, n_centres)

        POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP)

        ftree = generate_path_ftree_better(
            POINTS_INFO, MIX_PARAMS,
            length_cutoff=3,
            tuning_dist=0.02,
            tuning_strength=1,
            tuning_strength_diff=1.5,
            n_spanning_gap=N_SPANNING_GAP,
            n_slices_behind=0,
            n_slices_ahead=0
        )

        vars = list(ftree.get_variables())
        var_middle = vars[(len(vars) // 2)-1]

        traversal, run = ftree.run_max_quality_forward(var_middle)
        good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=4)
        # Get the top 3 best paths
        best_paths_start = {}
        for i in range(4):
            best = max(good_paths_start.keys(), key=lambda x: good_paths_start[x][1])
            best_paths_start[best] = good_paths_start[best]
            del good_paths_start[best]
            if not good_paths_start:
                break
        best_paths_info = calculate_good_paths(best_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

        bestest_mrf_ep = None
        for i, path_info in enumerate(best_paths_info):
            mrf_neb = neb.neb_mep(path_info, POINTS_INFO, MIX_PARAMS,
                                                n_spanning_point_gap=2, n_max_iterations=2500)
            mrf_ep = gaussian_field(mrf_neb, MIX_PARAMS)
            if bestest_mrf_ep is None or np.max(bestest_mrf_ep) > np.max(mrf_ep):
                bestest_mrf_ep = mrf_ep
        mrf_eps.append(bestest_mrf_ep)

        print(f'Running time {time.time() - start_time}')

        # Get the top 4 best paths
        # best_paths_info = []
        # for i in range(3):
        #     best = max(good_paths_info, key=lambda x: x['value'])
        #     good_paths_info.remove(best)
        #     best_paths_info.append(best)

        # fig1, ax1 = plt.subplots()
        # fig2, ax2 = plt.subplots()

        # plot_gaussian(MIX_PARAMS, fig1, ax1)
        # ax1.plot(*mrf_neb, label='MRF')

        # ax2.plot(mrf_ep, label='MRF')

        # for i, path_info in enumerate(good_paths_info):
        #     print(i, path_info['value'])
        #     lines = ax1.plot(*path_info['path'], label=f'q{i}={path_info["value"]}')
        #     neb_path_1 = neb.neb_mep(path_info, POINTS_INFO, MIX_PARAMS, k=2., n_spanning_point_gap=2, n_max_iterations=8000)
        #     ax1.plot(*neb_path_1, c=lines[0].get_color(), dashes=(2, 2))
        #     ax2.plot(gaussian_field(neb_path_1, MIX_PARAMS), c=lines[0].get_color(), label=i)


        old_path = np.linspace(MIX_PARAMS['minima_coords'][:, 0], MIX_PARAMS['minima_coords'][:, 1], mrf_neb.shape[1], axis=-1)
        new_path = neb.neb(old_path, MIX_PARAMS, n_max_iterations=2500)
        # lines = ax1.plot(*new_path, label='NEB')
        neb_ep = gaussian_field(new_path, MIX_PARAMS)
        neb_eps.append(neb_ep)
        # ax2.plot(neb_ep, label='NEB')

        # ax1.legend()
        # ax2.set_title('Energy profile of paths')
        # ax2.set_xlabel('Point along path')
        # ax2.set_ylabel('Energy')
        # ax2.legend()
        # fig1.show()
        # fig2.show()

    results.append((mrf_eps, neb_eps))

# breakdown_good_path(good_paths_info[3], ftree, quality_function, POINTS_INFO)

for dim, dim_r in zip(dims, results):
    fig, ax = plt.subplots()
    mrf_max = np.array([max(x) for x in dim_r[0]])
    neb_max = np.array([max(x) for x in dim_r[1]])
    print(f'{dim}D 0.2 better', np.sum(neb_max - mrf_max > 0.2)/len(dim_r[0]))
    print(f'{dim}D ~same', np.sum(neb_max - mrf_max > -0.1)/len(dim_r[0]))
    ax.hist(neb_max - mrf_max, density=True)
    ax.set_xlabel('NEB Max - MRF Max')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of difference of MRF to NEB ({dim}D) with multiselect')
    fig.show()

results_json = [
    [
        [runn.tolist() for runn in part]
        for part in dim
    ]
    for dim in results
]

from time import strftime
with open(f'energy_profiles/{"".join(str(dim) for dim in dims)}{strftime("%Y%m%dT%H%M%S")}.json', 'w') as f:
    import json
    json.dump(results_json, f)
# fig, ax = plt.subplots()
# dim = 4
# ax.plot(results[dim-2][0][0])
# ax.plot(results[dim-2][1][0])

# 2D 0.2 better 0.44
# 2D ~same 0.93
# 3D 0.2 better 0.47
# 3D ~same 0.98
# 4D 0.2 better 0.38
# 4D ~same 0.9
# 5D 0.2 better 0.34
# 5D ~same 0.89

# Multiselect version
# 2D 0.2 better 0.43
# 2D ~same 0.98
# 3D 0.2 better 0.72
# 3D ~same 1.0
# 4D 0.2 better 0.58
# 4D ~same 1.0
# 5D 0.2 better 0.3
# 5D ~same 1.0
# 6D 0.2 better 0.06
# 6D ~same 1.0
