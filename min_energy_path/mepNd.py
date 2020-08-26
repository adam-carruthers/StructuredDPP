import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from min_energy_path.path_helpers import generate_path_ftree_better
import min_energy_path.gaussian_params as mix_params
from min_energy_path.points_sphere import create_sphere_points
from min_energy_path import neb


logging.basicConfig(level=logging.INFO)

# First set up some constants we're going to need
N_SPANNING_GAP = 5


dims = [2, 3, 4, 5, 6, 7, 8]
labels = [f'{dim}D' for dim in dims]
label_pos = list(range(len(labels)))
times = []

for dim in dims:
    # Constants relating to the gaussian field
    MIX_PARAMS = mix_params.randomly_generated(dim, dim+2)

    start_time = time.time()

    POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP, shrink_in_direction=0.75)

    sphere_time_cuml = time.time() - start_time
    sphere_time = sphere_time_cuml

    ftree = generate_path_ftree_better(
        POINTS_INFO, MIX_PARAMS,
        length_cutoff=3,
        tuning_dist=0.01,
        tuning_strength=1,
        tuning_strength_diff=2,
        n_spanning_gap=N_SPANNING_GAP,
        n_slices_behind=1,
        n_slices_ahead=2
    )

    ftree_time_cuml = time.time() - start_time
    ftree_time = ftree_time_cuml - sphere_time_cuml

    # traversal, run = ftree.run_max_quality_forward(var_middle)
    # good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=50)
    # good_paths_info = calculate_good_paths(good_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

    tail_var = next(iter(ftree.levels[-1]))
    traversal, run = ftree.run_max_quality_forward(tail_var)

    forward_pass_cuml = time.time() - start_time
    forward_pass = forward_pass_cuml - ftree_time_cuml

    root_max_m, root_max_m_assignment = tail_var.calculate_max_message_assignment(run)
    logging.info(f'Max path has quality {root_max_m}, starting assigning')
    assignment = ftree.get_max_from_start_assignment(tail_var, root_max_m_assignment, traversal, run)

    path_indexes = [assignment[var] for var in ftree.get_variables()]
    path = np.array([
        POINTS_INFO['sphere'][:, path_index] for path_index in path_indexes
    ]).T

    assignment_time_cuml = time.time() - start_time
    assignment_time = assignment_time_cuml - forward_pass_cuml

    neb_path = neb.neb_mep({'path_indexes': path_indexes, 'path': path},
                           POINTS_INFO, MIX_PARAMS, n_spanning_point_gap=2)

    neb_time_cuml = time.time() - start_time
    neb_time = neb_time_cuml - assignment_time_cuml

    times.append([sphere_time, ftree_time, forward_pass, assignment_time, neb_time])

fig, ax = plt.subplots()

for i, time_type in enumerate(['Sphere', 'FTree', 'Forward', 'Assign', 'NEB']):
    ax.barh(label_pos, [x[i] for x in times], left=[sum(x[:i]) for x in times] if i else None, label=time_type)

ax.set_yticks(label_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel('Time (s)')
ax.set_title('Running time in different dimensions')
ax.legend()
