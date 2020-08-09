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
    grid_griddy = np.meshgrid(dir_component, *other_components)
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

    return {'sphere_before': sphere_before,
            'sphere': sphere,
            'sphere_index': np.arange(sphere.shape[1]),
            'n_total': n_total,
            'basis': basis,
            'minima_distance': minima_distance,
            'point_distance': point_distance,
            'n_overflow': n_overflow,
            'dir_component': dir_component,
            'root_index': root_index,
            'tail_index': tail_index,
            'root_dir_index': root_dir_index
            }


def plot_minima_and_circle(minima, scatter_with, gap_proportion=0.7):
    scatter_with.plot(*minima, 'rx-')
    midpoint = (minima[:, 0] + minima[:, 1]) / 2
    dist = scila.norm(minima[:, 0] - minima[:, 1])
    plt.gca().add_artist(plt.Circle(midpoint, dist / (2 * gap_proportion), color='r', fill=False))


def plot_scatter(scatter_points, scatter_with):
    scatter_with.scatter(*scatter_points, c='b', marker='o')
    if scatter_with == plt:
        plt.axis('equal')
    plt.show()


def plot_scatter_with_minima(scatter_points, minima, scatter_with, gap_proportion=0.7):
    plot_minima_and_circle(minima, scatter_with, gap_proportion)
    plot_scatter(scatter_points, scatter_with)


if __name__ == "__main__":
    # Plot a 2D grid with a number of different options

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)

    minima = np.array([
        [1, 0],
        [0, 1]
    ])
    gap_proportion = 0.7
    sphere_untransformed, sphere_transformed = create_sphere_points(minima, 8, gap_proportion, 1)

    plot_scatter_with_minima(sphere_transformed, minima, plt, gap_proportion)

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
