import numpy as np
from min_energy_path.gaussian_field import gaussian_field_for_quality
from structured_dpp.factor_tree import *


def generate_sphere_slice_path(intermediate_factor_quality, root_index, tail_index, root_dir_index, dir_component,
                               sphere_index, sphere_before, n_variables, n_slices_behind, n_slices_ahead):
    current_var = Variable((root_index,), name='RootVar0')
    nodes_to_add = [current_var]
    for i in range(1, n_variables):
        # Add transition factor
        transition_factor = Factor(intermediate_factor_quality,
                                   parent=current_var,
                                   name=f'Fac{i-1}-{i}')
        nodes_to_add.append(transition_factor)

        if i == n_variables - 1:  # Give the last variable only one possible position, the tail
            current_var = Variable((tail_index,),
                                   parent=transition_factor,
                                   name=f'TailVar{i}')
        else:
            # Sphere slice bounds
            slice_of_dir = dir_component[max(root_dir_index+i-n_slices_behind, 0):root_dir_index+i+1+n_slices_ahead]
            in_slice = (np.min(slice_of_dir) <= sphere_before[0, :]) & (sphere_before[0, :] <= np.max(slice_of_dir))

            current_var = Variable(sphere_index[in_slice].T,
                                   parent=transition_factor,
                                   name=f'Var{i}')
        nodes_to_add.append(current_var)

    ftree = FactorTree.create_from_connected_nodes(nodes_to_add)

    return ftree, nodes_to_add


def get_standard_factor(sphere, mix_mag, mix_sig, mix_centre, point_distance, length_cutoff,
                        tuning_dist, tuning_strength, tuning_strength_diff, tuning_grad, tuning_second_order,
                        order_2_step=0.2):
    """
    Returns the factor quality function given the input parameters.
    """
    @assignment_to_var_arguments
    def intermediate_factor_quality(idx0, idx1, return_breakdown=False):  # idx1 closer to the root
        if idx0 == idx1:
            return 0
        coords = sphere[:, [idx0, idx1]]

        gaussian_info = gaussian_field_for_quality(
            coords, mix_mag, mix_sig, mix_centre, point_distance, length_cutoff, order_2_step
        )
        if gaussian_info == 0:  # Returns 0 when length cutoff reached
            return 0

        pos0_strength, pos1_strength, mid_strength, second_order_guess, direction_length, orthog_grad_length = gaussian_info

        score = (
                # Distance score
                # Favor smaller distances
                # Give negative score to long distances
                - tuning_dist * direction_length / point_distance,
                # Strength score
                # Favor lower strengths
                # Give negative score to very positive strengths
                - tuning_strength * (pos0_strength + mid_strength + pos1_strength) / 3,
                # Strength diff quality
                # Penalise going upward
                # If pos0_strength (closer to the tail) is bigger than pos1_strength
                # then a negative value will be added to the score
                + tuning_strength_diff * min(0, pos1_strength - max(mid_strength, pos0_strength)),
                # Gradient score
                # Favor small tangential gradients in areas with a high second order derivative
                # Give negative score to very large orthogonal gradient lengths
                - tuning_grad * orthog_grad_length,
                # Second order score
                # Favor the path being at a minimum orthogonal to the path
                # This means that the two points orthogonal to the direction of the path
                # will have higher strengths than the midpoint
                + tuning_second_order * second_order_guess
        )

        if return_breakdown:
            return score

        return np.exp(sum(score))
    return intermediate_factor_quality


def get_good_path_start_samples(var, run, sphere_before, point_distance, n_per_group=3):
    """
    Takes var, which has had a max quality run performed on it, and returns a start_sample of the max quality paths.
    The start_sample is the max path that passes through one section of the variables allowed variables.
    """
    sample = {}
    for idx in var.allowed_values:
        group = tuple(sphere_before[1:, idx] / point_distance // n_per_group)
        value = var.outgoing_messages[run][None][idx]
        route_before = sample.get(group, None)
        if route_before is None or value > route_before[1]:
            sample[group] = (idx, value)
    return sample


def calculate_good_paths(start_sample, var, traversal, run, ftree):
    """
    Given a generated max path start_sample, calculate the paths to follow and return the path data.
    """
    return [
        [grp, start_val, path_value, ftree.get_max_from_start_assignment(var, start_val, traversal, run)]
        for grp, (start_val, path_value) in start_sample.items()
    ]


def breakdown_good_path(good_path, ftree: FactorTree, quality, sphere, len_breakdown=5):
    """
    Print to the console a breakdown of a good path and its quality breakdown
    """
    grp, good_max_idx, good_max, assignment = good_path

    print(f'''Examining path in group {grp}
q = {good_max}
------------''')
    total = [0]*len_breakdown
    for node in ftree.get_nodes():
        if isinstance(node, Factor):
            quality_breakdown = quality(node, assignment, return_breakdown=True)
            print(node, quality_breakdown)
            total = [x+y for x,y in zip(total, quality_breakdown)]
        else:
            print(node, sphere[:, assignment[node]])
    print('Overall', total)
