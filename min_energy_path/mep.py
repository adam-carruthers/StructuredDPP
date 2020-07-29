import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from structured_dpp.factor_tree import *
from min_energy_path.plot_gaussian import mix_mag, mix_sig, mix_centre, minima_coords, plot_gaussian
from min_energy_path.gaussian_field import gaussian_field, gaussian_field_grad
from min_energy_path.points_sphere import create_sphere_points, plot_scatter_with_minima


start_time = time.time()


# First set up some constants we're going to need
N_SPANNING_GAP = 15
N_VARIABLES = N_SPANNING_GAP

TUNING_STRENGTH = 1
TUNING_STRENGTH_DIFF = .75
TUNING_GRAD = 2
TUNING_DIST = .8

POINTS_INFO = create_sphere_points(minima_coords, N_SPANNING_GAP)
SPHERE_BEFORE = POINTS_INFO['sphere_before']
SPHERE = POINTS_INFO['sphere']
SPHERE_INDEX = np.arange(SPHERE.shape[1])
N_TOTAL = POINTS_INFO['n_total']
BASIS = POINTS_INFO['basis']
MINIMA_DISTANCE = POINTS_INFO['minima_distance']
POINT_DISTANCE = POINTS_INFO['point_distance']
N_OVERFLOW = POINTS_INFO['n_overflow']
DIR_COMPONENT = POINTS_INFO['dir_component']

# Find the minima indices
ROOT_INDEX, TAIL_INDEX = None, None
for i, point in enumerate(SPHERE.T):
    if np.all(point == minima_coords[:, 0]):
        ROOT_INDEX = i
    if np.all(point == minima_coords[:, 1]):
        TAIL_INDEX = i
if ROOT_INDEX is None or TAIL_INDEX is None:
    raise ValueError("Couldn't find root or tail index.")

# Plot the space we're exploring
plot_gaussian()
plot_scatter_with_minima(SPHERE, minima_coords, plt)


@assignment_to_var_arguments
def intermediate_factor_quality(idx1, idx2):
    if idx1 == idx2:
        return 0
    pos = SPHERE[:, [idx1, idx2]]
    pos1, pos2 = pos.T
    midpoint = (pos1 + pos2) / 2
    pos = np.concatenate((pos, midpoint[:, np.newaxis]), axis=1)

    direction = pos2 - pos1
    length = scila.norm(direction)
    if length >= 3 * POINT_DISTANCE:
        return 0
    direction_normed = direction/length
    dist_quality = np.exp(-TUNING_DIST*length/(2*POINT_DISTANCE))

    strength = gaussian_field(pos, mix_mag, mix_sig, mix_centre)
    strength_diff = strength[0] - strength[1]  # Strength 1 is closer to the root, as it is the parent
    strength_diff_quality = np.exp(-TUNING_STRENGTH_DIFF*strength_diff) if strength_diff > 0 else 1
    strength_quality = np.exp(-TUNING_STRENGTH*np.sum(strength)/3)

    grad = gaussian_field_grad(midpoint[:, np.newaxis], mix_mag, mix_sig, mix_centre)[:, 0]
    grad_perp = grad - np.dot(grad, direction_normed) * direction_normed
    grad_quality = np.exp(-TUNING_GRAD*scila.norm(grad_perp))
    return strength_quality*strength_diff_quality*dist_quality*grad_quality


def error_diversity(*args):
    raise Exception("This shouldn't run")


current_var = Variable((ROOT_INDEX,), name='RootVar0')
nodes_to_add = [current_var]
for i in range(1, N_VARIABLES):
    # Add transition factor
    transition_factor = SDPPFactor(intermediate_factor_quality,
                                   error_diversity,
                                   parent=current_var,
                                   name=f'Fac{i-1}-{i}')
    nodes_to_add.append(transition_factor)

    if i == N_VARIABLES - 1:
        current_var = Variable((TAIL_INDEX,),
                               parent=transition_factor,
                               name=f'TailVar{i}')
    else:
        # Sphere slice bounds
        slice_of_dir = DIR_COMPONENT[i-1:i+3]
        in_slice = (np.min(slice_of_dir) <= SPHERE_BEFORE[0, :]) & (SPHERE_BEFORE[0, :] <= np.max(slice_of_dir))

        current_var = Variable(SPHERE_INDEX[in_slice].T,
                               parent=transition_factor,
                               name=f'Var{i}')
    nodes_to_add.append(current_var)

ftree = SDPPFactorTree.create_from_connected_nodes(nodes_to_add)

assignments = ftree.sample_quality_only(10)


print(f'Running time {time.time() - start_time}')


plot_gaussian()
for assignment in assignments:
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    plt.plot(*points)

plt.show()
