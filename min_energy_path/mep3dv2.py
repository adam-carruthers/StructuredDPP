import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as scila
import time

from structured_dpp.factor_tree import *
from min_energy_path.gaussian_field import gaussian_field_for_quality
from min_energy_path.guassian_params import medium3d
from min_energy_path.points_sphere import create_sphere_points, plot_scatter_with_minima


start_time = time.time()


# First set up some constants we're going to need
N_SPANNING_GAP = 7
N_VARIABLES = N_SPANNING_GAP + 1

# Constants relating to the gaussian field
MIX_MAG, MIX_SIG, MIX_CENTRE, MINIMA_COORDS, XBOUNDS, YBOUNDS, ZBOUNDS = medium3d()

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_scatter_with_minima(SPHERE, MINIMA_COORDS, ax)

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

TUNING_DIST = .25
TUNING_STRENGTH = 1.5
TUNING_STRENGTH_DIFF = 1
TUNING_GRAD = 0.5
TUNING_SECOND_ORDER = 1
LENGTH_CUTOFF = 4

@assignment_to_var_arguments
def intermediate_factor_quality(idx0, idx1):  # idx1 closer to the root
    if idx0 == idx1:
        return 0
    coords = SPHERE[:, [idx0, idx1]]

    gaussian_info = gaussian_field_for_quality(coords, MIX_MAG, MIX_SIG, MIX_CENTRE, POINT_DISTANCE, LENGTH_CUTOFF)
    if gaussian_info == 0:  # Returns 0 when length cutoff reached
        return 0

    pos0_strength, pos1_strength, mid_strength, second_order_guess, direction_length, orthog_grad_length = gaussian_info

    score = (
        # Distance score
        # Favor smaller distances
        # Give negative score to long distances
        - TUNING_DIST * direction_length / POINT_DISTANCE
        # Strength score
        # Favor lower strengths
        # Give negative score to very positive strengths
        - TUNING_STRENGTH * (pos0_strength + mid_strength + pos1_strength) / 3
        # Strength diff quality
        # Penalise going upward
        # If pos0_strength (closer to the tail) is bigger than pos1_strength
        # then a negative value will be added to the score
        + TUNING_STRENGTH_DIFF * min(0, pos1_strength - max(mid_strength, pos0_strength))
        # Gradient score
        # Favor small tangential gradients in areas with a high second order derivative
        # Give negative score to very large orthogonal gradient lengths
        - TUNING_GRAD * orthog_grad_length * second_order_guess
        # Second order score
        # Favor the path being at a minimum orthogonal to the path
        # This means that the two points orthogonal to the direction of the path
        # will have higher strengths than the midpoint
        + TUNING_SECOND_ORDER * second_order_guess
    )

    return np.exp(score)


@assignment_to_var_arguments
def intermediate_factor_quality_breakdown(idx0, idx1):  # idx1 closer to the root
    if idx0 == idx1:
        return 0, 0, 0, 0, 0
    coords = SPHERE[:, [idx0, idx1]]

    gaussian_info = gaussian_field_for_quality(coords, MIX_MAG, MIX_SIG, MIX_CENTRE, POINT_DISTANCE, LENGTH_CUTOFF)
    if gaussian_info == 0:  # Returns 0 when length cutoff reached
        return 0

    pos0_strength, pos1_strength, mid_strength, orthog0_strength, orthog1_strength, direction_length, \
        orthog_grad_length = gaussian_info

    score = (
        # Distance score
        # Favor smaller distances
        # Give negative score to long distances
        np.exp(- TUNING_DIST * direction_length / POINT_DISTANCE),
        # Strength score
        # Favor lower strengths
        # Give negative score to very positive strengths
        np.exp(- TUNING_STRENGTH * (pos0_strength + mid_strength + pos1_strength) / 3),
        # Strength diff quality
        # Penalise going upward
        # If pos0_strength (closer to the tail) is bigger than pos1_strength
        # then a negative value will be added to the score
        np.exp(TUNING_STRENGTH_DIFF * min(0, pos1_strength - max(mid_strength, pos0_strength))),
        # Gradient score
        # Favor small tangential gradients
        # Give negative score to very large orthogonal gradient lengths
        np.exp(- TUNING_GRAD * orthog_grad_length),
        # Second order score
        # Favor the path being at a minimum orthogonal to the path
        # This means that the two points orthogonal to the direction of the path
        # will have higher strengths than the midpoint
        np.exp(TUNING_SECOND_ORDER * (orthog0_strength - 2*mid_strength + orthog1_strength))
    )

    return score


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
        slice_of_dir = DIR_COMPONENT[max(ROOT_DIR_INDEX+i-1, 0):ROOT_DIR_INDEX+i+2]
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

index_of_middle = len(nodes_to_add) // 2
var5 = nodes_to_add[index_of_middle - (index_of_middle % 2) - 2]

traversal, run = ftree.run_max_quality_forward(var5)
good_max_samples = get_good_max_samples(var5, run, 4)
assignments = [ftree.get_max_from_start_assignment(var5, good_max_idx, traversal, run)
               for good_max_idx, good_max in good_max_samples.values()]

print(f'Running time {time.time() - start_time}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(*MIX_CENTRE, c=MIX_MAG, s=40*MIX_SIG)

good_maxs = list(zip(assignments, good_max_samples.items()))

for assignment, (grp, (good_max_idx, good_max)) in good_maxs:
    points = np.array([
        SPHERE[:, assignment[var]] for var in ftree.get_variables()
    ]).T
    ax.plot(*points, label=f'{grp}')
    print(grp, good_max)


plt.legend()
plt.show()

def examine_good_max(index):
    print(f'Examining group {good_maxs[4][1][0]}\n-------')

    assignment = good_maxs[index][0]
    quality_breakdown = [1, 1, 1, 1, 1]
    for node in ftree.get_nodes():
         if isinstance(node, SDPPFactor):
             fac_qual_breakdown = intermediate_factor_quality_breakdown(node, assignment)
             print(node, fac_qual_breakdown)
             quality_breakdown = [q1 * q2 for q1, q2 in zip(quality_breakdown, fac_qual_breakdown)]
         else:
             print(node, SPHERE[:, assignment[node]])
    print('Overall', quality_breakdown)
