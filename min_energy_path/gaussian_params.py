# noinspection PyUnresolvedReferences
"""
Specification for mix_params, a dictionary, is in gaussian_field.py
"""
import numpy as np


def hashtag_blocked():
    mix_centre = np.array([[0,  2, 2,   3.5, 5.5, 6,   4.5],
                           [1,  1, -1,  0,   2,   0,   -2]])
    return {
        'magnitude': np.array([-1, 1, 1.5, 0.8, 1, 0.8, -1.2]),
        'sigma': np.array([1,  1, 1,   1.5, 1, 0.5, 1]),
        'centre': mix_centre,
        'minima_coords': mix_centre[:, [0, -1]],
        'xbounds': (-2, 7),
        'ybounds': (-5, 4)
    }

def starter():
    mix_centre = np.array([
        [0, 0],
        [2, 1],
        [3, -1.1],
        [4, 0],
        [5, 0]
    ]).T
    return {
        'magnitude': np.array([-1, 2, 1, 1.5, -1]),
        'sigma': np.array([2, 0.5, 0.5, 0.5, 1.5]),
        'centre': mix_centre,
        'minima_coords': mix_centre[:, [0, -1]],
        'xbounds': (-2, 7),
        'ybounds': (-5, 4)
    }

def basic3d():
    mix_centre = np.array([[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2]])
    return {
        'magnitude': np.array([-1, 1, -1]),
        'sigma': np.array([1, 1.5, 1]),
        'centre': mix_centre,
        'minima_coords': mix_centre[:, [0, -1]],
        'xbounds': (-1, 3),
        'ybounds': (-1, 3),
        'zbounds': (-1, 3)
    }

def medium3d():
    return {
        'magnitude': np.array([-1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, -1]),
        'sigma': np.ones(10),
        'centre': np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                           [0, -1, -1, -1, 0, 0, 0, 1, 1, 0],
                           [0, -1, 0, 1, -1, 0, 1, -1, 0, 0]]),
        'minima_coords': np.array([
            [-0.8, 0, 0],
            [2.8, 0, 0]
        ]).T,
        'xbounds': (-1, 3),
        'ybounds': (-1, 3),
        'zbounds': (-2, 2)
    }

def complex2d():
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
    return {
        'magnitude': np.array([-1]*2 + [1]*6),
        'sigma': np.ones(8)*0.5,
        'centre': mix_centre,
        'minima_coords': mix_centre[:, [0, 1]],
        'xbounds': (-1, 9),
        'ybounds': (-1, 5)
    }

def shallow():
    return {
        'magnitude': np.array([-20]+[-5]*3),
        'sigma': np.array([40, 10, 10, 10]),
        'centre': np.array([
            [100, 100],
            [100, 120],
            [75, 75],
            [110, 60]
        ]).T,
        'minima_coords': np.array([
            [100, 117],
            [110, 60]
        ]).T,
        'xbounds': (50, 150),
        'ybounds': (50, 150)
    }

def even_simplerer():
    return {
        'magnitude': np.array([-1, 2, -1]),
        'sigma': np.array([1, 1, 1]),
        'centre': np.array([
            [0, 0],
            [1, 1],
            [2, 0]
        ]).T,
        'minima_coords': np.array([
            [-0.45, -0.45],
            [2.55, -0.45]
        ]).T,
        'xbounds': (-2, 4),
        'ybounds': (-3, 3)
    }

if __name__ == '__main__':
    # Quick code to see what a 2D path looks like
    from min_energy_path.gaussian_field import plot_gaussian
    gaussian_params = shallow()
    plot_gaussian(gaussian_params)
