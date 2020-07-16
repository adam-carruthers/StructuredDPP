from unittest import TestCase
from structured_dpp.semiring import *
import numpy as np


# noinspection DuplicatedCode
class TestOrder2MatrixSemiring(TestCase):
    def test_add(self):
        # Basic variables
        x = Order2MatrixSemiring(1, np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
        y = Order2MatrixSemiring(1, np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]))

        # Normal addition
        xpy_ans = Order2MatrixSemiring(2, np.array([5, 7, 9]), np.array([11, 13, 15]), np.array([
                [2, 2, 4],
                [4, 6, 6],
                [8, 8, 10]
            ]))
        self.assertEqual(x + y, xpy_ans, "Addition didn't work")
        self.assertEqual(y + x, x + y, "Addition doesn't work the other way around")

        # Adding zero
        self.assertEqual(x + 0, x, "Additon of zero shouldn't change semiring")
        self.assertEqual(0 + x, x, "Addition of zero doesn't work the other way around")
        self.assertEqual(x.zero_like() + x, x, "Addition of zero like x failed")

        # Checking commutivity
        self.assertEqual((x + y) + 0, xpy_ans, "3 term addition failed")
        self.assertEqual(x + (0 + y), xpy_ans, "3 term addition failed")

        # Adding one
        xp1_ans = Order2MatrixSemiring(2, np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
        self.assertEqual(x + 1, xp1_ans, "Addition of 1 failed")
        self.assertEqual(x + x.one_like(), xp1_ans, "Addition of 1 like x failed")

    def test_eq(self):
        x = Order2MatrixSemiring(99, np.array([-22, 45]), np.array([37, -28]), np.array([[-8, -9], [24.5, 28.12]]))
        self.assertEqual(
            x,
            Order2MatrixSemiring(99, np.array([-22, 45]), np.array([37, -28]), np.array([[-8, -9], [24.5, 28.12]])),
            "Equality operator does not work."
        )
        x + 1 + 0
        x * 1
        self.assertEqual(
            x,
            Order2MatrixSemiring(99, np.array([-22, 45]), np.array([37, -28]), np.array([[-8, -9], [24.5, 28.12]])),
            "Operations changed x when x supposed to be immutable!"
        )
        self.assertNotEqual(
            x,
            Order2MatrixSemiring(98, np.array([-22, 45]), np.array([37, -28]), np.array([[-8, -9], [24.5, 28.12]])),
            "!= operator does not work."
        )
        self.assertNotEqual(
            x,
            Order2MatrixSemiring(99, np.array([-22, 45]), np.array([37, -28]), np.array([[-8, -9], [24.5, 50]])),
            "!= operator does not work."
        )

    def test_mul(self):
        # Basic variables
        x = Order2MatrixSemiring(0.5, np.array([2, 3]), np.array([4, 6]), np.array([
            [1, 2],
            [4, 5],
        ]))
        y = Order2MatrixSemiring(4, np.array([5, 6]), np.array([7, 8]), np.array([
            [1, 0],
            [0, 1]
        ]))

        # Multiplying by 1
        self.assertEqual(x * 1, x, "Multiplying by 1 doesn't give back original")
        self.assertEqual(1 * x, x, "Multiplying by 1 the other way around doesn't give back original")
        self.assertEqual(x.one_like() * x, x, "Multiplying by 1 like x doesn't give back original")
        self.assertEqual(y * y.one_like(), y, "Multiplying by 1 like y the other way round doesn't give back original")

        # Multiplying by 0
        self.assertEqual(x * 0, 0, "Multiplying by 0 doesn't give back 0")
        self.assertEqual(0 * x, x.zero_like(), "Multiplying by 0 the other way around doesn't give back 0")
        self.assertEqual(x.zero_like() * x, 0, "Multiplying by 0 like x doesn't give back 0")
        self.assertEqual(y * y.zero_like(), 0, "Multiplying by 0 like y the other way round doesn't give back 0")

        # Normal multiplication
        xmy_ans = Order2MatrixSemiring(2, np.array([2.5 + 8, 3 + 12]), np.array([3.5 + 16, 4 + 24]),
                                       np.array([[4.5 + 2*7 + 5*4, 8 + 2*8 + 5*6],
                                               [16 + 3*7 + 6*4, 20.5 + 3*8 + 6*6]]))
        self.assertEqual(x*y, xmy_ans, "Multiplication failed")

    def test_zero_one_like(self):
        x = Order2MatrixSemiring(0.5, np.array([2, 3]), np.array([4, 6]), np.array([
            [1, 2],
            [4, 5],
        ]))
        sem_zero_x = Order2MatrixSemiring(0, np.array([0, 0]), np.array([0, 0]), np.array([[0, 0], [0, 0]]))
        sem_one_x = Order2MatrixSemiring(1, np.array([0, 0]), np.array([0, 0]), np.array([[0, 0], [0, 0]]))

        self.assertEqual(x.zero_like(), sem_zero_x, "Zero generated is incorrect.")
        self.assertEqual(x.one_like(), sem_one_x, "One generated is incorrect.")

        y = Order2MatrixSemiring(1, np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]))
        sem_zero_y = Order2MatrixSemiring(0, np.array([0, 0, 0]), np.array([0, 0, 0]),
                                          np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        sem_one_y = Order2MatrixSemiring(1, np.array([0, 0, 0]), np.array([0, 0, 0]),
                                         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        self.assertEqual(y.one_like(), sem_one_y, "One generated is incorrect.")
        self.assertEqual(y.zero_like(), sem_zero_y, "Zero generated is incorrect.")


class TestOrder2VectSemiring(TestCase):
    def test_basic_example(self):
        x = Order2VectSemiring(0.5, np.array([2, 3, 4]), np.array([4, 6, 8]), np.array([1, 2, 3]))
        y = Order2VectSemiring(4, np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([1, 0, 1]))
        z = Order2VectSemiring(2, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))

        self.assertEqual(
            1 * (x + y),
            Order2VectSemiring(4.5, np.array([6, 8, 10]), np.array([11, 14, 17]), np.array([2, 2, 4]))
        )

        self.assertEqual(
            x * (y * 1 + 0),
            Order2VectSemiring(2, np.array([2+8, 2.5+12, 3+16]), np.array([3.5+16, 4+24, 4.5+32]),
                               np.array([.5+4+14+16, 8+24+30, .5+12+36+48]))
        )

        self.assertEqual(
            y * z,
            Order2VectSemiring(8, np.array([8, 10, 12]), np.array([14, 16, 18]), np.array([2, 0, 2]))
        )
