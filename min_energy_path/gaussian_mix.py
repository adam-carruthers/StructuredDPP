import numpy as np


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
