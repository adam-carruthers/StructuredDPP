import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as scila
import time

from structured_dpp.factor_tree import *
from min_energy_path.gaussian_field import gaussian_field, gaussian_field_grad, plot_gaussian
from min_energy_path.guassian_params import starter
from min_energy_path.points_sphere import create_sphere_points, plot_scatter_with_minima


start_time = time.time()


# First set up some constants we're going to need
N_SPANNING_GAP = 10
N_VARIABLES = N_SPANNING_GAP + 2

TUNING_STRENGTH = 1.5
TUNING_STRENGTH_DIFF = 1
TUNING_GRAD = 1
TUNING_DIST = .25
LENGTH_CUTOFF = 4

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

# Plot the space we're exploring
plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)
plot_scatter_with_minima(SPHERE, MINIMA_COORDS, plt)

# Find the minima indices
ROOT_INDEX, TAIL_INDEX = None, None
for i, point in enumerate(SPHERE.T):
    if np.allclose(point, MINIMA_COORDS[:, 0]):
        ROOT_INDEX = i
    if np.allclose(point, MINIMA_COORDS[:, 1]):
        TAIL_INDEX = i
if None in [ROOT_INDEX, TAIL_INDEX]:
    raise ValueError("Couldn't find root or tail index.")
ROOT_DIR_INDEX = np.where(DIR_COMPONENT == SPHERE_BEFORE[0, ROOT_INDEX])[0][0]


@assignment_to_var_arguments
def intermediate_factor_quality(idx1, idx2):  # idx2 closer to the root
    if idx2 == TAIL_INDEX:
        if idx1 == TAIL_INDEX:
            return np.e
        else:
            return 0
    if idx1 == idx2:
        return 0
    pos = SPHERE[:, [idx1, idx2]]
    pos1, pos2 = pos.T
    midpoint = (pos1 + pos2) / 2
    pos = np.concatenate((pos, midpoint[:, np.newaxis]), axis=1)

    direction = pos2 - pos1
    length = scila.norm(direction)
    if length >= LENGTH_CUTOFF * POINT_DISTANCE:
        return 0
    direction_normed = direction/length
    dist_quality = np.exp(-TUNING_DIST*length/(2*POINT_DISTANCE))

    strength = gaussian_field(pos, MIX_MAG, MIX_SIG, MIX_CENTRE)
    strength_diff = strength[0] - strength[1]  # Strength 1 is closer to the root, as it is the parent
    strength_diff_quality = np.exp(-TUNING_STRENGTH_DIFF*strength_diff) if strength_diff > 0 else 1
    strength_quality = np.exp(-TUNING_STRENGTH*np.sum(strength)/3)

    grad = gaussian_field_grad(midpoint[:, np.newaxis], MIX_MAG, MIX_SIG, MIX_CENTRE)[:, 0]
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
        slice_of_dir = DIR_COMPONENT[max(ROOT_DIR_INDEX+i-2, 0):ROOT_DIR_INDEX+i+2]
        in_slice = (np.min(slice_of_dir) <= SPHERE_BEFORE[0, :]) & (SPHERE_BEFORE[0, :] <= np.max(slice_of_dir))

        current_var = Variable(SPHERE_INDEX[in_slice].T,
                               parent=transition_factor,
                               name=f'Var{i}')
    nodes_to_add.append(current_var)

ftree = SDPPFactorTree.create_from_connected_nodes(nodes_to_add)

def get_good_max_samples(var: Variable, run, n_per_group=3):
    groups = {}
    for idx in var.allowed_values:
        group = tuple(SPHERE_BEFORE[1:, idx] / POINT_DISTANCE // n_per_group)
        value = var.outgoing_messages[run][None][idx]
        route_before = groups.get(group, None)
        if route_before is None or value > route_before[1]:
            groups[group] = (idx, value)
    return groups

var5 = nodes_to_add[10]

traversal, run = ftree.run_max_quality_forward(var5)
good_max_samples = get_good_max_samples(var5, run, 3)
assignments = [ftree.get_max_from_start_assignment(var5, good_max_idx, traversal, run)
               for good_max_idx, good_max in good_max_samples.values()]

print(f'Running time {time.time() - start_time}')

plot_gaussian(MIX_MAG, MIX_SIG, MIX_CENTRE, XBOUNDS, YBOUNDS)

var5_max_beliefs: dict = var5.outgoing_messages[MaxProductRun()][None]
belief_pos = np.array([SPHERE[:, idx] for idx in var5_max_beliefs.keys()]).T
belief_vals = [1 if val > 0.001 else 0 for val in var5_max_beliefs.values()]
plt.scatter(*belief_pos, c=belief_vals)
for idx in var5_max_beliefs.keys():
    plt.gca().annotate(str(idx), SPHERE[:, idx])

for assignment, (good_max_idx, good_max) in zip(assignments, good_max_samples.values()):
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    plt.plot(*points, label=f'q={good_max:.3e}')

plt.legend()
plt.show()
