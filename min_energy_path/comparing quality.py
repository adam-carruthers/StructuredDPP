import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from min_energy_path.gaussian_field import plot_gaussian
import min_energy_path.gaussian_params as mix_params
from min_energy_path.points_sphere import create_sphere_points
from min_energy_path.path_helpers import (get_standard_transition_quality_function, get_good_path_start_samples,
                                          calculate_good_paths, breakdown_good_path, generate_path_ftree_better)
from min_energy_path import neb

start_time = time.time()

# First set up some constants we're going to need
N_SPANNING_GAP = 10
N_VARIABLES = N_SPANNING_GAP + 2

# Constants relating to the gaussian field
MIX_PARAMS = mix_params.even_simplerer()

POINTS_INFO = create_sphere_points(MIX_PARAMS['minima_coords'], N_SPANNING_GAP)

ftree = generate_path_ftree_better(
    POINTS_INFO, MIX_PARAMS,
    length_cutoff=3,
    tuning_dist=0.02,
    tuning_strength=1,
    tuning_strength_diff=1.5,
    n_spanning_gap=N_SPANNING_GAP,
    n_slices_behind=1,
    n_slices_ahead=2
)

quality_function = get_standard_transition_quality_function(
    POINTS_INFO, MIX_PARAMS,
    length_cutoff=3,
    tuning_dist=0.02,
    tuning_strength=1,
    tuning_strength_diff=1.5,
    tuning_grad=0,
    tuning_second_order=0
)

fact = next(iter(ftree.get_factors()))
transition_qualities = fact.transition_qualities

root = fact.parent
child = next(iter(fact.children))

print(quality_function(fact, {root: 11, child:5}, return_breakdown=True))
