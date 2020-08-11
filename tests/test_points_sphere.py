from unittest import TestCase
import numpy as np
from min_energy_path.points_sphere import *


class TestPointsSphere(TestCase):
    def test_get_nearby_sphere_indexes(self):
        minima = np.array([
            [1, 0],
            [2, 0]
        ]).T
        points_info = create_sphere_points(minima, 8)
        indexes_near = get_nearby_sphere_indexes(14, 2, points_info, slices_behind=1)
        indexes_near.sort()
        self.assertListEqual(
            indexes_near,
            [5, 6, 7, 8, 12, 13, 15, 16, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34]
        )

        indexes_near_2 = get_nearby_sphere_indexes(54, 1, points_info, return_center=True, return_lower=False)
        self.assertListEqual(
            indexes_near_2,
            [54, 55, 62, 63, 64]
        )
