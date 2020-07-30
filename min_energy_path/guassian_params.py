import numpy as np


def hashtag_blocked():
    mix_mag =     np.array([-1, 1, 1.5, 0.8, 1,   0.8, -1.2])
    mix_sig =     np.array([1,  1, 1,   1.5, 1,   0.5, 1])
    mix_centre = np.array([[0,  2, 2,   3.5, 5.5, 6,   4.5],
                           [1,  1, -1,  0,   2,   0,   -2]])
    minima_coords = mix_centre[:, [0, -1]]
    xbounds = (-2, 7)
    ybounds = (-5, 4)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds


def starter():
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
    xbounds = (-2, 7)
    ybounds = (-5, 4)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds
