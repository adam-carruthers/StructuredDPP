import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as scila


def transform_coords(coords, mix_centre):
    """
    Take in column vector coords and transform to be centred for each gaussian mix
    :param coords:
        Column vectors (d dimensions, n points)
    :param mix_centre:
        Column vectors (d dimensions, m gaussians)
    :return:
        Matrix (d dimensions, n points, m gaussians)
    """
    return coords[:, :, np.newaxis] - mix_centre[:, np.newaxis, :]


def gaussian_field(coords, mix_mag, mix_sig, mix_centre):
    """
    Calculate the strength of the gaussian mix field at a set of coordinates
    :param coords:
        Matrix shape (d dimensions, n points)
    :param mix_magnitude:
        Vector shape (m gaussians,)
    :param mix_sigma:
        Vector shape (m gaussians,)
    :param mix_centre:
        Matrix shape (d dimensions, m gaussians)
    :return:
        Vector shape (n points,) with the field strength at that point
    """
    return np.sum(_field_strength(transform_coords(coords, mix_centre), mix_mag, mix_sig), axis=1)


def _field_strength(coords_transformed, mix_magnitude, mix_sigma):
    return mix_magnitude[np.newaxis, :] * np.exp(
        np.sum(-coords_transformed**2, axis=0) / (2*mix_sigma[np.newaxis, :]**2)
    )


def gaussian_grad(coords, mix_magnitude, mix_sigma, mix_centre):
    """
    Calculate the grad of a gaussian mixture field at a set of coordinates
    :param coords:
        Matrix shape (d dimensions, n points)
    :param mix_magnitude:
        Vector shape (m gaussians,)
    :param mix_sigma:
        Vector shape (m gaussians,)
    :param mix_centre:
        Matrix shape (d dimensions, m gaussians)
    :return:
        Matrix shape (d dimensions, n points) with column vectors of the gradient in each dimension at each point
    """
    coords_transformed = transform_coords(coords, mix_centre)
    field_strength = _field_strength(coords_transformed, mix_magnitude, mix_sigma)
    return _gaussian_grad(coords_transformed, field_strength, mix_sigma)


def _gaussian_grad(coords_transformed, field_strength, mix_sigma):
    """
    Work out the gradient having already done some prior calculations
    :param coords_transformed:
        Each point as a column vector transformed to the centre of each gaussian mix
        Multiarray (d dimensions, n points, m gaussians)
    :param field_strength:
        The strength of the gaussian field for each point for each gaussian field
        Matrix (n points, m gaussians)
    :param mix_sigma:
        The sigma value for each gaussian
        Vector (m gaussians,)
    :return:
    """
    scale = -field_strength / mix_sigma[np.newaxis, :] ** 2
    return np.sum(scale[np.newaxis, :, :] * coords_transformed, axis=2)


def gaussian_field_for_quality(coords, mix_mag, mix_sig, mix_centre, point_distance, length_cutoff, order_2_step=0.2):
    direction = coords[:, 1] - coords[:, 0]
    direction_length = scila.norm(direction)
    if direction_length > length_cutoff * point_distance:
        return 0
    midpoint = (coords[:, [0]] + coords[:, [1]]) / 2
    coords = np.concatenate([coords, midpoint], axis=1)

    # Find the field strength of the points on the line
    coords_transformed = transform_coords(coords, mix_centre)
    field_strength = _field_strength(coords_transformed, mix_mag, mix_sig)
    pos0_strength, pos1_strength, mid_strength = np.sum(field_strength, axis=1)

    # Find the magnitude of the gradient perpendicular to the direction of the path
    grad = _gaussian_grad(coords_transformed[:, [2], :], field_strength[[2], :], mix_sig)[:, 0]
    orthog_grad = grad - np.dot(grad, direction) * direction / direction_length**2
    orthog_grad_length = scila.norm(orthog_grad)

    # Get a good vector perpendicular to the direction of the path,
    # preferably in the highest gradient direction
    # not possible for small gradients, or gradients in the direction of the path
    if orthog_grad_length >= 1e-8:
        orthog_direction = orthog_grad / orthog_grad_length
    else:
        min_dir = np.argmin(direction)
        basis = np.zeros(coords.shape[0])
        basis[min_dir] = 1
        orthog_direction = basis - np.dot(basis, direction) * basis
        orthog_direction /= scila.norm(orthog_direction)

    # Take small step either direction from the midpoint in the orthogonal direction
    orthog_points = midpoint + np.array([[-1, 1]]) * point_distance * order_2_step
    orthog0_strength, orthog1_strength = gaussian_field(orthog_points, mix_mag, mix_sig, mix_centre)
    # Use the finite difference method of estimating the second order derivative
    second_order_guess = (orthog1_strength - 2*mid_strength + orthog0_strength) / (point_distance * order_2_step)**2

    return pos0_strength, pos1_strength, mid_strength, second_order_guess, direction_length, orthog_grad_length





def plot_gaussian(mix_mag, mix_sig, mix_centre, xbounds, ybounds):
    # Plot the scalar field
    # First create an x, y grid
    x = np.linspace(*xbounds, 100)
    y = np.linspace(*ybounds, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
    coords = np.array([x_flat,
                       y_flat])

    # Work out z
    z_flat = gaussian_field(coords, mix_mag, mix_sig, mix_centre)
    z_grid = z_flat.reshape(x_grid.shape)


    # Set up the color map
    cmap = plt.cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=np.min(z_flat), vmax=np.max(z_flat))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot everything
    fig, ax = plt.subplots()
    plt.pcolormesh(x_grid, y_grid, z_grid, cmap=cmap)
    fig.colorbar(sm)


def get_gaussian_slice(xbounds, ybounds, zbounds, xstep, ystep, zstep, basis, mix_mag, mix_sig, mix_centre):
    x = np.arange(*xbounds, xstep)
    y = np.arange(*ybounds, ystep)
    z = np.arange(*zbounds, zstep)
    X, Y, Z = np.meshgrid(x, y, z)
    coords = np.array([X.flatten(), Y.flatten(), Z.flatten])
    F = gaussian_field(basis @ coords, mix_mag, mix_sig, mix_centre).reshape(X.shape)



