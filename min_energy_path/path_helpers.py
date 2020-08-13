import numpy as np

from min_energy_path.mep_ftree import MEPFactor, MEPVariable
from min_energy_path.gaussian_field import gaussian_field_for_quality, gaussian_field_for_better_quality

from structured_dpp.factor_tree import *


def generate_path_ftree(quality_function, points_info, n_spanning_gap, n_slices_behind, n_slices_ahead):
    current_var = Variable((points_info['root_index'],), name='RootVar0')
    nodes_to_add = [current_var]
    for i in range(n_spanning_gap+1):
        # Add transition factor
        transition_factor = Factor(quality_function,
                                   parent=current_var,
                                   name=f'Fac{i}-{i+1}')
        nodes_to_add.append(transition_factor)

        if i == n_spanning_gap:  # Give the last variable only one possible position, the tail
            current_var = Variable((points_info['tail_index'],),
                                   parent=transition_factor,
                                   name=f'TailVar{i+1}')
        else:
            # Sphere slice bounds
            slice_of_dir = points_info['dir_component'][
                max(points_info['root_dir_index']+i-n_slices_behind, 0):points_info['root_dir_index']+i+1+n_slices_ahead
            ]
            in_slice = (np.min(slice_of_dir) <= points_info['sphere_before'][0, :]) & (points_info['sphere_before'][0, :] <= np.max(slice_of_dir))

            current_var = Variable(points_info['sphere_index'][in_slice].T,
                                   parent=transition_factor,
                                   name=f'Var{i+1}')
        nodes_to_add.append(current_var)

    ftree = FactorTree.create_from_connected_nodes(nodes_to_add)

    return ftree


def generate_path_ftree_better(points_info, mix_params,
                               # Parameters relating to transition quality
                               length_cutoff,
                               tuning_dist, tuning_strength, tuning_strength_diff,
                               # Parameters relating to variables and slicing
                               n_spanning_gap, n_slices_behind=1, n_slices_ahead=2):
    transition_qualities = generate_transition_qualities(
        points_info, mix_params, length_cutoff, tuning_dist, tuning_strength, tuning_strength_diff, n_slices_behind,
        n_slices_ahead
    )

    current_var = Variable((points_info['root_index'],), name='RootVar0')
    nodes_to_add = [current_var]

    for i in range(n_spanning_gap+1):
        # Add transition factor
        transition_factor = MEPFactor(transition_qualities, length_cutoff, n_slices_behind, n_slices_ahead, points_info,
                                      parent=current_var, name=f'Fac{i}-{i+1}')
        nodes_to_add.append(transition_factor)

        if i == n_spanning_gap:  # Give the last variable only one possible position, the tail
            current_var = Variable((points_info['tail_index'],),
                                   parent=transition_factor,
                                   name=f'TailVar{i+1}')
        else:
            # Sphere slice bounds
            slice_start = max(points_info['root_dir_index']+i-n_slices_behind, 0)
            slice_end = points_info['root_dir_index']+i+1+n_slices_ahead
            slice_of_dir = points_info['dir_component'][
                slice_start:slice_end
            ]
            in_slice = (np.min(slice_of_dir) <= points_info['sphere_before'][0, :]) & (points_info['sphere_before'][0, :] <= np.max(slice_of_dir))

            current_var = MEPVariable(points_info['sphere_index'][in_slice].T,
                                      slice_start, slice_end,
                                      parent=transition_factor,
                                      name=f'Var{i+1}')
        nodes_to_add.append(current_var)

    ftree = FactorTree.create_from_connected_nodes(nodes_to_add)

    return ftree


def get_standard_transition_quality_function(points_info, mix_params, length_cutoff,
                                             tuning_dist, tuning_strength, tuning_strength_diff, tuning_grad, tuning_second_order,
                                             order_2_step=0.2):
    """
    Returns the factor quality_function function given the input parameters.
    """
    @assignment_to_var_arguments
    def intermediate_factor_quality(idx0, idx1, return_breakdown=False):  # idx1 closer to the root
        if idx0 == idx1:
            return 0
        coords = points_info['sphere'][:, [idx0, idx1]]

        gaussian_info = gaussian_field_for_quality(
            coords, mix_params, points_info['point_distance'], length_cutoff, order_2_step
        )
        if gaussian_info == 0:  # Returns 0 when length cutoff reached
            return 0

        pos0_strength, pos1_strength, mid_strength, second_order_guess, direction_length, orthog_grad_length = gaussian_info

        score = (
                # Distance score
                # Favor smaller distances
                # Give negative score to long distances
                - tuning_dist * direction_length / points_info['point_distance'],
                # Strength score
                # Favor lower strengths
                # Give negative score to very positive strengths
                - tuning_strength * (
                    ((pos0_strength + mid_strength) / 2 - mix_params['min_minima_strength'])
                    / mix_params['max_line_strength_diff']
                ),
                # Strength diff quality_function
                # Penalise going upward
                # If pos0_strength (closer to the tail) is bigger than pos1_strength
                # then a negative value will be added to the score
                + tuning_strength_diff * (
                    min(0, pos1_strength - max(mid_strength, pos0_strength))
                    / mix_params['max_line_strength_diff']
                ),
                # Gradient score
                # Favor small tangential gradients in areas with a high second order derivative
                # Give negative score to very large orthogonal gradient lengths
                - tuning_grad * orthog_grad_length * np.exp(tuning_second_order * second_order_guess),
                # Second order score
                # Favor the path_guess being at a minimum orthogonal to the path_guess
                # This means that the two points orthogonal to the direction of the path_guess
                # will have higher strengths than the midpoint
                # Only give this advantage if the path has slipped to the minimum point (small orthog grad)
                + tuning_second_order * second_order_guess * np.exp(-tuning_grad*orthog_grad_length)
        )

        if return_breakdown:
            return score

        return np.exp(sum(score))
    return intermediate_factor_quality


def get_good_path_start_samples(var, run, points_info, n_per_group=3):
    """
    Takes var, which has had a max quality_function run performed on it, and returns a start_sample of the max quality_function paths.
    The start_sample is the max path_guess that passes through one section of the variables allowed variables.
    """
    sample = {}
    for idx in var.allowed_values:
        group = tuple(points_info['sphere_before'][1:, idx] / points_info['point_distance'] // n_per_group)
        value = var.outgoing_messages[run][None][idx]
        route_before = sample.get(group, None)
        if route_before is None or value > route_before[1]:
            sample[group] = (idx, value)
    return sample


def calculate_good_paths(start_sample, var, traversal, run, ftree, points_info):
    """
    Given a generated max path_guess start_sample, calculate the paths to follow and return the path_guess data.
    """
    path_infos = []
    for grp, (start_val, path_value) in start_sample.items():
        assignment = ftree.get_max_from_start_assignment(var, start_val, traversal, run)
        path_indexes = [assignment[var] for var in ftree.get_variables()]
        path = np.array([
            points_info['sphere'][:, path_index] for path_index in path_indexes
        ]).T
        path_infos.append({
            'group': grp,
            'start_idx': start_val,
            'value': path_value,
            'assignment': assignment,
            'path': path,
            'path_indexes': path_indexes
        })
    return path_infos


def breakdown_good_path(good_path, ftree: FactorTree, quality_function, points_info, len_breakdown=5):
    """
    Print to the console a breakdown of a good path_guess and its quality_function breakdown
    """

    print(f'''Examining path_guess in group {good_path['group']}
q = {good_path['value']}
Dist, Strength, Strength Diff, Grad, Second Order
------------''')
    total = [0]*len_breakdown
    for node in ftree.get_nodes():
        if isinstance(node, Factor):
            quality_breakdown = quality_function(node, good_path['assignment'], return_breakdown=True)
            quality = np.exp(np.sum(quality_breakdown))
            print(node, quality_breakdown, quality)
            total = [x+y for x,y in zip(total, quality_breakdown)]
        else:
            print(node, good_path['assignment'][node], points_info['sphere'][:, good_path['assignment'][node]])
    print('Overall', total)


def generate_transition_qualities(points_info, mix_params,
                                  # Parameters for the quality
                                  length_cutoff,
                                  tuning_dist, tuning_strength, tuning_strength_diff,  # not doing grad qualities
                                  # Parameters for the path variables
                                  n_slices_behind, n_slices_ahead):
    # Step 1 - Work out all the possible transition qualities
    # First, we work out which variables we need to calculate transitions from
    min_dir_index = max(points_info['root_dir_index']-n_slices_behind, 0)
    max_dir_index = points_info['tail_dir_index']+1+n_slices_ahead
    to_calculate_from = points_info['spherey_index'][min_dir_index:max_dir_index, ...]

    # Then actually calculate, for each "from" point, the quality to each possible "to" point
    # from is rootwards, to is leafwards
    # transition_qualities[rootwards][leafwards]
    transition_qualities = {}

    for from_pos, fromm in np.ndenumerate(to_calculate_from):
        if fromm == -1:
            continue
        to_slices_behind = from_pos[0] - n_slices_behind - n_slices_ahead + 1
        to_slices_ahead = from_pos[0] + n_slices_behind + n_slices_ahead + 2
        to_calculate_idx = points_info['spherey_index'][
            (slice(max(to_slices_behind, min_dir_index), min(to_slices_ahead, max_dir_index)),) +
            tuple(
                slice(max(from_dim_pos-length_cutoff, 0), from_dim_pos+1+length_cutoff)
                for from_dim_pos in from_pos[1:]
            )
        ]
        to_calculate_idx = to_calculate_idx[~np.isin(to_calculate_idx, [-1, fromm])]
        directions_length, to_calculate_idx, from_strength, midpoint_strengths, to_strengths = \
            gaussian_field_for_better_quality(
                points_info['sphere'][:, [fromm]], points_info['sphere'][:, to_calculate_idx], to_calculate_idx,
                mix_params, length_cutoff, points_info['point_distance']
            )
        to_qualities = np.exp(
            - tuning_dist * directions_length / points_info['point_distance']
            - tuning_strength * (
                ((midpoint_strengths + to_strengths) / 2 - mix_params['min_minima_strength'])
                / mix_params['max_line_strength_diff']
            )
            - tuning_strength_diff * (
                np.maximum(np.maximum(midpoint_strengths, to_strengths) - from_strength, 0)
                / mix_params['max_line_strength_diff']
            )
        )
        transition_qualities[fromm] = dict(zip(to_calculate_idx, to_qualities))

    return transition_qualities
