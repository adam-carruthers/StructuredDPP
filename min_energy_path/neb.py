"""
A file to implement NEB.

Credit to https://github.com/adamfrees/NudgedElasticBand
I copied his algorithm
"""
import numpy as np
import scipy.linalg as scila
import logging

from min_energy_path.gaussian_field import gaussian_grad


logger = logging.getLogger(__name__)


def neb(path_guess, mix_params, n_iterations=3000, k=1., time_step=1.e-2):
    """
    Performs the NEB algorithm
    :param np.ndarray path_guess:
        Matrix (d dimensions, n points) of the points on the initial path guess
    :param mix_params:
        The generated dictionary of gaussian mix params
    :param n_iterations:
        Number of iterations to perform on the path
    :param k:
        The spring force component
    :param time_step:
        The size of the step to take during an interation
    :return:
    """
    logger.info('Starting neb run')

    path = path_guess.copy()

    velocity = np.zeros((path.shape[0], path.shape[1]-2))
    old_force = np.zeros_like(velocity)

    for i in range(n_iterations):
        # Calculate the tangent at each point
        # Except the endpoints
        tangents = path[:, 2:] - path[:, :-2]
        tangents /= scila.norm(tangents, axis=0, keepdims=True)

        # Calculate the distance between the points
        transitions = path[:, 1:] - path[:, :-1]
        point_distances = scila.norm(transitions, axis=0, keepdims=True)

        # Spring force component!
        spring_component = k*(point_distances[:, 1:] - point_distances[:, :-1])*tangents

        # Tangential gradient component
        gradients = gaussian_grad(path[:, 1:-1], mix_params)
        orth_grad_component = gradients - np.sum(gradients * tangents, axis=0, keepdims=True)*tangents

        # Apply forces using gradient
        force = spring_component - orth_grad_component
        force_velocity = velocity * force
        if np.sum(force_velocity) > 0:
            velocity = force_velocity * force / scila.norm(force)**2
            velocity += (old_force+force) / 2.
        else:
            velocity = (old_force + force) / 2.

        path[:, 1:-1] += velocity * time_step

        old_force = force

    return path


def neb_mep(mepath_info, points_info, mix_params, n_spanning_point_gap=3, n_iterations=3000, k=1., time_step=1.e-2):
    """
    Performs the NEB algorithm on the MEP generated path
    :param mepath_info:
        The generated path_info dictionary
    :param points_info:
        The generated point_info dictionary
    :param n_spanning_point_gap:
        The number of NEB points to span the MEP point gap
    :param mix_params:
        The generated dictionary of gaussian mix params
    :param n_iterations:
        Number of iterations to perform on the path
    :param k:
        The spring force component
    :param time_step:
        The size of the step to take during an interation
    :return:
    """
    # First check if the path ever crosses the same point twice
    # If it does, delete it!
    new_path_points = []
    i = 0
    path_indexes = mepath_info['path_indexes']
    while i < len(path_indexes):
        new_path_points.append(i)
        for j in range(i+1, len(path_indexes)):
            if path_indexes[i] == path_indexes[j]:
                i = j  # Set i to j, skipping points in between
        i += 1

    new_mep_path = mepath_info['path'][:, new_path_points]

    neb_start_path = []
    for i in range(new_mep_path.shape[1]-1):
        p0 = new_mep_path[:, i]
        p1 = new_mep_path[:, i+1]
        dist = scila.norm(p1 - p0) / points_info['point_distance']
        n_points = int(np.floor((dist * n_spanning_point_gap) + 1e-4))
        points = np.linspace(p0, p1, n_points, endpoint=False, axis=-1)
        points += ((p1-p0)/(2*n_points))[:, np.newaxis]  # Shift the points so that they are equidistant from both end points
        neb_start_path.append(points)

    neb_start_path = np.concatenate(neb_start_path, axis=1)

    return neb(neb_start_path, mix_params, n_iterations, k, time_step)


if __name__ == '__main__':
    from min_energy_path.gaussian_params import even_simplerer
    from min_energy_path.gaussian_field import plot_gaussian
    import matplotlib.pyplot as plt

    _mix_params = even_simplerer()
    old_path = np.linspace(_mix_params['minima_coords'][:, 0], _mix_params['minima_coords'][:, 1], 15, axis=-1)
    new_path = neb(old_path, _mix_params)

    plot_gaussian(_mix_params)
    plt.plot(*old_path, 'bo-')
    plt.plot(*new_path, 'ro-')
