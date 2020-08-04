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
        [1, 0.5, 3, -1.1],
        [1.5, 0.5, 4, 0],
        [-1, 1.5, 5, 0]
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

def basic3d():
    mix_mag = np.array([-1, 1, -1])
    mix_sig = np.array([1, 1.5, 1])
    mix_centre = np.array([[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2]])
    minima_coords = mix_centre[:, [0, -1]]
    xbounds = (-1, 3)
    ybounds = (-1, 3)
    zbounds = (-1, 3)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds, zbounds

def medium3d():
    mix_mag = np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1, -1])
    mix_sig = np.ones_like(mix_mag)
    mix_centre = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                           [0, -1, -1, -1, 0, 0, 0, 1, 1, 0],
                           [0, -1, 0, 1, -1, 0, 1, -1, 0, 0]])
    minima_coords = mix_centre[:, [0, -1]]
    xbounds = (-1, 3)
    ybounds = (-1, 3)
    zbounds = (-2, 2)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds, zbounds

def complex2d():
    mix_mag = np.array([-1]*2 + [1]*6)
    mix_sig = np.ones_like(mix_mag)*0.5
    mix_centre = np.array([
        [1, 2],
        [7, 2.5],
        [1, 0.5],
        [2, 3.5],
        [3, 1.5],
        [5.5, 4],
        [5.5, 2.5],
        [5.5, .5]
    ]).T
    minima_coords = mix_centre[:, [0, 1]]
    xbounds = (-1, 9)
    ybounds = (-1, 5)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds

def shallow():
    mix_mag = np.array([-20]+[-5]*3)
    mix_sig = np.array([40, 10, 10, 10])
    mix_centre = np.array([
        [100, 100],
        [100, 120],
        [75, 75],
        [115, 75]
    ]).T
    minima_coords = np.array([
        [100, 117],
        [113, 79]
    ]).T
    xbounds = (50, 150)
    ybounds = (50, 150)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds

def even_simplerer():
    mix_mag = np.array([-1, 2, -1])
    mix_sig = np.array([1, 1, 1])
    mix_centre = np.array([
        [0, 0],
        [1, 1],
        [2, 0]
    ]).T
    minima_coords = np.array([
        [-0.45, -0.45],
        [2.55, -0.45]
    ]).T
    xbounds = (-2, 4)
    ybounds = (-3, 3)
    return mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds

if __name__ == '__main__':
    # Quick code to see what a 2D path looks like
    from min_energy_path.gaussian_field import plot_gaussian
    mix_mag, mix_sig, mix_centre, minima_coords, xbounds, ybounds = even_simplerer()
    plot_gaussian(mix_mag, mix_sig, mix_centre, xbounds, ybounds)
