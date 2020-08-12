import numpy as np
import scipy.linalg as scila
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings


def create_sphere_basis(minima):
    """
    Creates an orthonormal basis matrix where the first basis vector has the same direction as the vector from the
    first minima to the other minima
    :param minima:
        Matrix size (d dimensions, 2) column vectors of the two minima
    :return:
        Matrix of orthonormal basis vectors
    """
    dimensions = minima.shape[0]
    minima_delta = minima[:, 1] - minima[:, 0]
    minima_direction = minima_delta/scila.norm(minima_delta)
    pre_basis = np.eye(dimensions) - np.outer(minima_direction, minima_direction)
    basis = scila.orth(pre_basis)
    assert basis.shape == (dimensions, dimensions-1)
    assert np.allclose(minima_direction[np.newaxis, :] @ basis, 0)
    return np.concatenate((minima_direction[:, np.newaxis], basis), axis=1)


def create_sphere_points(minima, n_spanning_gap, gap_proportion=0.7, shrink_in_direction=1.0):
    """
    Creates the sphere of points between two minima
    :param minima:
        Matrix size (d dimensions, 2) column vectors of the two minima
    :param n_spanning_gap:
        The number of points between the two minima including the two minima
    :param gap_proportion:
        The proportion of the diameter of the sphere that is spanned by the two minima
    :param shrink_in_direction:
        The extent to which the sphere will be shrunk in all axis except the one in the direction of the minima
    :return:
    """
    # Useful quantities
    # Get quantities relating to the position of the minima and the vector between them
    dimensions = minima.shape[0]
    minima_delta = minima[:, 1] - minima[:, 0]
    minima_distance = scila.norm(minima_delta)
    minima_direction = minima_delta / minima_distance
    point_distance = minima_distance/(n_spanning_gap - 1)
    sphere_radius = minima_distance/(2*gap_proportion)

    # Calculate the total number of points, which is wider than the points spanning the gap
    # The points spanning the gap only span {gap_proportion} of the diameter
    n_total = (n_spanning_gap - 1) / gap_proportion
    # There must be an even number of points outside the gap, an equal number each side
    n_total -= (n_total - n_spanning_gap) % 2
    if n_total < n_spanning_gap + 2:
        warnings.warn("n_spanning_gap and gap_proportion mean that there aren't enough points "
                      "for some to go behind the minima")
    n_overflow = (n_total - n_spanning_gap)/2
    n_horizontal = np.ceil(shrink_in_direction*n_total/2)

    # Where is the first point located?
    first_point_pos = minima[:, 0] - minima_direction * point_distance * n_overflow

    # Create a grid
    dir_component = np.arange(n_total)*point_distance
    other_component = np.arange(-n_horizontal, n_horizontal+1)*point_distance
    other_components = np.tile(other_component, (dimensions-1, 1))
    grid_griddy = np.meshgrid(dir_component, *other_components, indexing='ij')
    grid_flattened = np.array([component.flatten() for component in grid_griddy])  # This turns the grid into column vectors

    measurement_grid = grid_flattened.copy()
    measurement_grid[0, :] -= np.max(dir_component)/2
    measurement_grid[1:, :] /= shrink_in_direction
    in_sphere = scila.norm(measurement_grid, axis=0) <= sphere_radius
    sphere_before = grid_flattened[:, in_sphere]

    basis = create_sphere_basis(minima)
    sphere = first_point_pos[:, np.newaxis] + basis @ sphere_before

    # Find the minima indices
    root_index, tail_index = None, None
    for i, point in enumerate(sphere.T):
        if np.allclose(point, minima[:, 0]):
            root_index = i
        if np.allclose(point, minima[:, 1]):
            tail_index = i
    if None in [root_index, tail_index]:
        raise ValueError("Couldn't find root or tail index.")
    root_dir_index = np.where(dir_component == sphere_before[0, root_index])[0][0]
    tail_dir_index = np.where(dir_component == sphere_before[0, tail_index])[0][0]

    # Sphere index stuff
    sphere_index = np.arange(sphere.shape[1])
    spherey_index = np.ones(grid_flattened.shape[1]) * -1
    spherey_index[in_sphere] = sphere_index
    spherey_index = spherey_index.reshape(grid_griddy[0].shape).astype(int)
    spherey_index_index = np.arange(grid_flattened.shape[1])[in_sphere].copy()

    return {'sphere_before': sphere_before,
            'sphere': sphere,
            'sphere_index': sphere_index,
            'spherey_index': spherey_index,  # A grid whos elements are the 2D index of the elements in the sphere array
            'spherey_index_index': spherey_index_index,  # Array sphere index => grid index of spherey flattened
            'n_total': n_total,
            'basis': basis,
            'minima_distance': minima_distance,
            'point_distance': point_distance,
            'n_overflow': n_overflow,
            'dir_component': dir_component,
            'root_index': root_index,
            'tail_index': tail_index,
            'root_dir_index': root_dir_index,
            'tail_dir_index': tail_dir_index
            }


def get_nearby_sphere_indexes(center_index, n_around, points_info,
                              slices_behind=None, slices_ahead=None, return_center=False, return_lower=True,
                              min_dir_index=0, max_dir_index=None):
    """
    Takes in a sphere_index (that being the index of the column of the matrix of points in the sphere).
    It outputs a list of sphere_indexes in the grid n_around the point.
    For example:
    - The sphere is 2D
    - * is the point corresponding to the sphere_index given
    - x is points not in the sphere
    - Points labelled o are returned
    - . (dot) is not
    n_around = 2
    . . . . x x x
    . o o o o x x
    . o o o o x x
    . o o * o o x
    . o o o o o x
    . o o o o o x
    . . . . . x x
    :param center_index:
        The index of the column in the sphere matrix corresponding to the point to return
    :param n_around:
        How many steps to take in each direction around the grid
    :param points_info:
        The points_info dictionary, see function create_sphere_points
    :param slices_behind:
        In the first dimension, normally where slices are taken, how many slices behind to go.
        If none just the same as n_around
    :param slices_ahead:
        In the first dimension, normally where slices are taken, how many slices ahead to go.
        If none just the same as n_around
    :param bool return_center:
        Return the index of the center (the one passed to the function)
    :param bool return_lower:
        If true filter out indices lower than the center.
    :param min_dir_index:
        The minimum slice layer for which points can be returned
    :param max_dir_index:
        The upper value for the slice layer for which points can be returned.
        Keep in mind this means layer with index max_dir_index won't be returned,
        whereas max_dir_index-1 can be.
    :return:
        List of indices in this area
    """
    if slices_ahead is None:
        slices_ahead = n_around
    if slices_behind is None:
        slices_behind = n_around
    if not return_lower:
        slices_behind = 0

    # Get the 3D index of the center in the spherey index
    c_spherey_idx_idx = points_info['spherey_index_index'][center_index]
    c_unraveled_idx = np.unravel_index(c_spherey_idx_idx, points_info['spherey_index'].shape)

    # Slice a grid around the center in the spherey index
    indices_to_scan = points_info['spherey_index'][
        (
            slice(
                max(c_unraveled_idx[0]-slices_behind, min_dir_index),
                (
                    c_unraveled_idx[0]+1+slices_ahead
                    if max_dir_index is None else
                    min(c_unraveled_idx[0]+1+slices_ahead, max_dir_index)
                )
            ),
        ) + tuple(
            slice(max(dim_index-n_around, 0), dim_index+1+n_around)
            for dim_index in c_unraveled_idx[1:]
        )
    ]
    return [
        sphere_index
        for sphere_index in indices_to_scan.flatten()
        if sphere_index >= 0
        and (return_center or sphere_index != center_index)
        and (return_lower or sphere_index >= center_index)
    ]


if __name__ == "__main__":
    # Plot a 2D grid with a number of different options

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)

    my_minima = np.array([
        [1, 0],
        [2, 0]
    ]).T
    points_info = create_sphere_points(my_minima, 8)

    plt.plot(*my_minima, 'r-')
    plt.scatter(*points_info['sphere'], c='b')
    for i, point in zip(points_info['sphere_index'], points_info['sphere'].T):
        plt.annotate(i, point)

    indexes_near = get_nearby_sphere_indexes(57, 3, points_info, slices_behind=1, slices_ahead=6)
    plt.scatter(*points_info['sphere'][:, indexes_near], c='r', zorder=5)

#     basis_3d = create_sphere_basis(np.array([
#         [0, 0],
#         [0, 2],
#         [0, 1]
#     ]))
#     for basis_vect in basis_3d.T:
#         ax.plot([0, basis_vect[0]], [0, basis_vect[1]], [0, basis_vect[2]])
#
#     ax.set_xlim3d(-1, 1)
#     ax.set_ylim3d(-1, 1)
#     ax.set_zlim3d(-1, 1)
#
#     plt.show()
