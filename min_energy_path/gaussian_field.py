import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as scila


def gaussian_field(coords, mix_magnitude, mix_sigma, mix_centre):
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
    coords = coords.T  # The formula below uses the coordinates as row vectors
    mix_centre = mix_centre.T
    return np.sum(mix_magnitude[np.newaxis, :] * np.exp(
        np.sum(-(coords[:, :, np.newaxis] - mix_centre.T[np.newaxis, :, :])**2, axis=1) / mix_sigma[np.newaxis, :]**2
    ), axis=1)


def gaussian_field_grad(coords, mix_magnitude, mix_sigma, mix_centre):
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
    coords = coords.T  # The formula below uses the coordinates as row vectors
    mix_centre = mix_centre.T
    coords_transformed = coords[:, :, np.newaxis] - mix_centre.T[np.newaxis, :, :]
    scale = mix_magnitude[np.newaxis, :] * -2 / mix_sigma[np.newaxis, :]**2 * np.exp(
        -np.sum(coords_transformed**2, axis=1) / mix_sigma**2
    )
    grad = np.sum(scale[:, np.newaxis, :] * coords_transformed, axis=2)
    return grad.T


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

