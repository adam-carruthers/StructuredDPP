# noinspection PyUnresolvedReferences
"""
Specification for mix_params, a dictionary

:key mag:
    Vector shape (m gaussians,)
:key sigma:
    Vector shape (m gaussians,)
:key centre:
    Matrix shape (d dimensions, m gaussians)
:key xbounds:
    Vector or tuple [lower bound, upper bound] - Optional
:key ybounds:
    Vector or tuple [lower bound, upper bound] - Optional
:key zbounds:
    Vector or tuple [lower bound, upper bound] - Optional
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as scila
import functools


def transform_coords(coords, mix_params):
    """
    Take in column vector coords and transform to be centred for each gaussian mix
    :param coords:
        Column vectors (d dimensions, n points)
    :param mix_params:
        See docstring for gaussian_field.py
    :return:
        Matrix (d dimensions, n points, m gaussians)
    """
    return coords[:, :, np.newaxis] - mix_params['centre'][:, np.newaxis, :]


def gaussian_field(coords, mix_params):
    """
    Calculate the strength of the gaussian mix field at a set of coordinates
    :param coords:
        Matrix shape (d dimensions, n points)
    :param mix_params:
        See docstring for gaussian_field.py
    :return:
        Vector shape (n points,) with the field strength at that point
    """
    return np.sum(_field_strength(transform_coords(coords, mix_params), mix_params), axis=1)


def _field_strength(coords_transformed, mix_params):
    """

    :param coords_transformed:
        Matrix shape (d dimensions, n points, m gaussians)
    :param mix_params:
        See docstring for gaussian_field.py
    :return:
        Matrix (n points, m gaussians)
    """
    return mix_params['magnitude'][np.newaxis, :] * np.exp(
        np.sum(-coords_transformed**2, axis=0) / (2*mix_params['sigma'][np.newaxis, :]**2)
    )


def gaussian_grad(coords, mix_params):
    """
    Calculate the grad of a gaussian mixture field at a set of coordinates
    :param coords:
        Matrix shape (d dimensions, n points)
    :param mix_params:
        See docstring for gaussian_field.py
    :return:
        Matrix shape (d dimensions, n points) with column vectors of the gradient in each dimension at each point
    """
    coords_transformed = transform_coords(coords, mix_params)
    field_strength = _field_strength(coords_transformed, mix_params)
    return _gaussian_grad(coords_transformed, field_strength, mix_params)


def _gaussian_grad(coords_transformed, field_strength, mix_params):
    """
    Work out the gradient having already done some prior calculations
    :param coords_transformed:
        Each point as a column vector transformed to the centre of each gaussian mix
        Multiarray (d dimensions, n points, m gaussians)
    :param field_strength:
        The strength of the gaussian field for each point for each gaussian field
        Matrix (n points, m gaussians)
    :param mix_params:
        See docstring for gaussian_field.py
    :return:
    """
    scale = -field_strength / mix_params['sigma'][np.newaxis, :] ** 2
    return np.sum(scale[np.newaxis, :, :] * coords_transformed, axis=2)


def gaussian_field_for_quality(coords, mix_params, point_distance, length_cutoff, order_2_step=0.2):
    direction = coords[:, 1] - coords[:, 0]
    direction_length = scila.norm(direction)
    if direction_length > length_cutoff * point_distance:
        return 0
    midpoint = (coords[:, [0]] + coords[:, [1]]) / 2
    coords = np.concatenate([coords, midpoint], axis=1)

    # Find the field strength of the points on the line
    coords_transformed = transform_coords(coords, mix_params)
    field_strength = _field_strength(coords_transformed, mix_params)
    pos0_strength, pos1_strength, mid_strength = np.sum(field_strength, axis=1)

    # Find the magnitude of the gradient perpendicular to the direction of the path_guess
    grad = _gaussian_grad(coords_transformed[:, [2], :], field_strength[[2], :], mix_params)[:, 0]
    orthog_grad = grad - np.dot(grad, direction) * direction / direction_length**2
    orthog_grad_length = scila.norm(orthog_grad)

    # Get a good vector perpendicular to the direction of the path_guess,
    # preferably in the highest gradient direction
    # not possible for small gradients, or gradients in the direction of the path_guess
    if orthog_grad_length >= 1e-8:
        orthog_direction = orthog_grad / orthog_grad_length
    else:
        min_dir = np.argmin(direction)
        basis = np.zeros(coords.shape[0])
        basis[min_dir] = 1
        orthog_direction = basis - np.dot(basis, direction) * basis
        orthog_direction /= scila.norm(orthog_direction)

    # Take small step either direction from the midpoint in the orthogonal direction
    orthog_points = midpoint + np.array([[-1, 1]]) * point_distance * order_2_step * orthog_direction[:, np.newaxis]
    orthog0_strength, orthog1_strength = gaussian_field(orthog_points, mix_params)
    # Use the finite difference method of estimating the second order derivative
    second_order_guess = (orthog1_strength - 2*mid_strength + orthog0_strength) / (point_distance * order_2_step)**2

    return pos0_strength, pos1_strength, mid_strength, second_order_guess, direction_length, orthog_grad_length


def gaussian_field_for_better_quality(from_coord, to_coords, to_coord_indices, mix_params, length_cutoff,
                                      point_distance):
    directions = to_coords - from_coord
    directions_length = scila.norm(directions, axis=0)

    # filter out elements too far away
    close_enough = directions_length / point_distance <= length_cutoff
    to_coords = to_coords[:, close_enough]
    to_coord_indices = to_coord_indices[close_enough]

    directions_length = directions_length[close_enough]

    # find the midpoint and join all the points together
    midpoints = (to_coords + from_coord) / 2

    # get the strength of the gaussian field
    from_strength = gaussian_field(from_coord, mix_params)[0]
    midpoint_strengths = gaussian_field(midpoints, mix_params)
    to_strengths = gaussian_field(to_coords, mix_params)

    return directions_length, to_coord_indices, from_strength, midpoint_strengths, to_strengths


def get_mix_params_info_decorator(n_iterations=1000, learning_rate=0.01, n_linspace=25):
    def find_mix_info_decorator(params_function):
        @functools.wraps(params_function)
        def params_with_info():
            mix_params = params_function()

            # First gradient descent to the actual minima
            minima_positions = mix_params['minima_guess']
            for i in range(n_iterations):
                minima_positions = minima_positions - learning_rate * gaussian_grad(minima_positions, mix_params)
            mix_params['minima_coords'] = minima_positions
            # Find the minima strength there
            mix_params['min_minima_strength'] = np.min(gaussian_field(minima_positions, mix_params))

            # Secondly scan the direct line between the two minima
            line = np.linspace(*minima_positions.T, n_linspace, axis=-1)
            mix_params['max_line_strength'] = np.max(gaussian_field(line, mix_params))
            mix_params['max_line_strength_diff'] = mix_params['max_line_strength'] - mix_params['min_minima_strength']
            return mix_params
        return params_with_info
    return find_mix_info_decorator


def plot_gaussian(mix_params):
    # Plot the scalar field
    # First create an x, y grid
    x = np.linspace(*mix_params['xbounds'], 100)
    y = np.linspace(*mix_params['ybounds'], 100)
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
    coords = np.array([x_flat,
                       y_flat])

    # Work out z
    z_flat = gaussian_field(coords, mix_params)
    z_grid = z_flat.reshape(x_grid.shape)


    # Set up the color map
    cmap = plt.cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=np.min(z_flat), vmax=np.max(z_flat))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot everything
    fig, ax = plt.subplots()
    plt.pcolormesh(x_grid, y_grid, z_grid, cmap=cmap)
    fig.colorbar(sm)
    ax.axis('equal')


def get_gaussian_slice(xstep, ystep, zstep, basis, mix_params):
    x = np.arange(*xbounds, xstep)
    y = np.arange(*ybounds, ystep)
    z = np.arange(*zbounds, zstep)
    X, Y, Z = np.meshgrid(x, y, z)
    coords = np.array([X.flatten(), Y.flatten(), Z.flatten])
    F = gaussian_field(basis @ coords, mix_mag, mix_sig, mix_centre).reshape(X.shape)
    # This function is unfinished
    # It aims to animate a field in 3D



