import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as scila

from min_energy_path.gaussian_field import gaussian_field, gaussian_grad


# First create the scalar field we're navigating
# The scalar field is simply the sum of A*exp(-((x-x0)**2 + (y-y0)**2)/sigma**2)
gaussian_mix_params = np.array([
    [-1, 2, 0, 0],  # A, sigma, x0, y0
    [2, 0.5, 2, 1],
    [1, 1, 3, -1.5],
    [1.5, 0.5, 4, 0],
    [-1, 1, 5, 0]
])
mix_mag = gaussian_mix_params[:, 0]
mix_sig = gaussian_mix_params[:, 1]
mix_centre = gaussian_mix_params[:, 2:].T  # Use column vectors

minima_coords = np.array([
    [0, 5],
    [0, 0]
])

def plot_gaussian(with_arrows=False):
    # Plot the scalar field
    # First create an x, y grid
    x = np.linspace(-2, 7, 100)
    y = np.linspace(-5, 4, 100)
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

    if with_arrows:
        # Work out the gradient to plot
        arrow_coords = np.array([[1, -1],
                                 [4.5, 0],
                                 [2, 2]]).T  # Use column vectors
        grad = gaussian_grad(arrow_coords, mix_mag, mix_sig, mix_centre)
        grad /= scila.norm(grad, axis=0)[np.newaxis, :] * 1.5
        for (arrow_x, arrow_y), (dx, dy) in zip(arrow_coords.T, grad.T):
            plt.arrow(arrow_x, arrow_y, dx, dy, head_width=0.2, head_length=0.1)

    # Plot the circle
    ax.add_artist(plt.Circle((2.5, 0), 5/(2*0.7), color='r', fill=False))

if __name__ == '__main__':
    plot_gaussian(True)
    plt.show()
