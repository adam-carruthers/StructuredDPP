import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from min_energy_path.path_helpers import generate_path_ftree_better
import min_energy_path.gaussian_params as mix_params
from min_energy_path.points_sphere import create_sphere_points
from min_energy_path.gaussian_field import gaussian_field
from min_energy_path import neb


logging.basicConfig(level=logging.INFO)

# First set up some constants we're going to need
N_SPANNING_GAP = 6

# Constants relating to the gaussian field
MIX_PARAMS = mix_params.randomly_generated(3, 4)

POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP, shrink_in_direction=0.75)

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

# traversal, run = ftree.run_max_quality_forward(var_middle)
# good_paths_start = get_good_path_start_samples(var_middle, run, POINTS_INFO, n_per_group=50)
# good_paths_info = calculate_good_paths(good_paths_start, var_middle, traversal, run, ftree, POINTS_INFO)

tail_var = next(iter(ftree.levels[-1]))
traversal, run = ftree.run_max_quality_forward(tail_var)

root_max_m, root_max_m_assignment = tail_var.calculate_max_message_assignment(run)
logging.info(f'Max path has quality {root_max_m}, starting assigning')
assignment = ftree.get_max_from_start_assignment(tail_var, root_max_m_assignment, traversal, run)

path_indexes = [assignment[var] for var in ftree.get_variables()]
path = np.array([
    POINTS_INFO['sphere'][:, path_index] for path_index in path_indexes
]).T

neb_path = neb.neb_mep({'path_indexes': path_indexes, 'path': path},
                       POINTS_INFO, MIX_PARAMS, n_spanning_point_gap=2, n_max_iterations=10**4)

straight_path = np.linspace(MIX_PARAMS['minima_coords'][:, 0], MIX_PARAMS['minima_coords'][:, 1], neb_path.shape[1], axis=-1)
naive_neb = neb.neb(straight_path, MIX_PARAMS, n_max_iterations=10**4)

plt.plot(gaussian_field(neb_path, MIX_PARAMS), label='Best MRF path')
plt.plot(gaussian_field(naive_neb, MIX_PARAMS), label='Naive NEB path')

ax = plt.gca()
ax.set_title('Energy profile of paths')
ax.set_xlabel('Point along path')
ax.set_ylabel('Energy')
ax.legend()

plt.show()
