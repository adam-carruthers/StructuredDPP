import numpy as np
import scipy.linalg as scila
import matplotlib.pyplot as plt

from min_energy_path.points_sphere import create_sphere_points


def generate_transition_qualities(points_info,
                                  # Parameters for the quality
                                  length_cutoff,
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
    n_total = 0

    for from_pos, fromm in np.ndenumerate(to_calculate_from):
        if fromm == -1:
            continue
        to_slices_behind = min_dir_index + from_pos[0] - n_slices_behind - n_slices_ahead + 1
        to_slices_ahead = min_dir_index + from_pos[0] + n_slices_behind + n_slices_ahead + 2
        to_calculate_idx = points_info['spherey_index'][
            (slice(max(to_slices_behind, min_dir_index), min(to_slices_ahead, max_dir_index)),) +
            tuple(
                slice(max(from_dim_pos-length_cutoff, 0), from_dim_pos+1+length_cutoff)
                for from_dim_pos in from_pos[1:]
            )
        ]
        to_calculate_idx = to_calculate_idx[~np.isin(to_calculate_idx, [-1, fromm])]
        directions = points_info['sphere'][:, to_calculate_idx] - points_info['sphere'][:, [fromm]]
        directions_length = scila.norm(directions, axis=0)

        n_total += np.sum(directions_length / points_info['point_distance'] <= length_cutoff)*2 + 1
    return n_total, to_calculate_from.size


if __name__ == '__main__':
    x = np.arange(2, 9)
    y_sphere_points = []
    y_transitions = []
    for dim in x:
        print(dim)
        points_info = create_sphere_points(np.array([[0, 1]] + [[0, 0]]*(dim-1)), 4, 0.7, 1)
        y_sphere_points.append(points_info['sphere'].shape[1])
        n_transitions, n_from = generate_transition_qualities(points_info, 3, 1, 1)
        y_transitions.append(n_transitions)

    fig, ax1 = plt.subplots()

    ax1.set_title('Changing dimensions')
    ax1.set_xlabel('dimensions')
    ax1.set_ylabel('Calls of field strength function', c='r')
    ax1.plot(x, y_transitions, c='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of points in sphere', c='b')
    ax2.plot(x, y_sphere_points, c='b')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.tight_layout()
    plt.show()

    # points_info = create_sphere_points(np.array([[0, 1]] + [[0, 0]]), 50)
    # generate_transition_qualities(points_info, 4, 1, 2)
