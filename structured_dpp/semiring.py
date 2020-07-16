import numpy as np
from collections import namedtuple


class Order2MatrixSemiring(namedtuple('Order2MatrixSemiring', ['p', 'phi', 'psi', 'C'])):
    """
    A class for a 2nd order vectorised semiring
    q = number
    phi = D-dimensional vector
    psi = D-dimensional vector
    C = DxD Matrix
    """
    __slots__ = ()  # Keeps memory low by stopping instance dictionary being created

    @property
    def D(self):
        return self.phi.shape[0]

    def one_like(self):
        D = self.D
        return Order2MatrixSemiring(1, np.zeros(D), np.zeros(D), np.zeros((D, D)))

    def zero_like(self):
        D = self.D
        return Order2MatrixSemiring(0, np.zeros(D), np.zeros(D), np.zeros((D, D)))

    def __add__(self, other):
        if other == 0:
            return self
        if other == 1:
            return self._replace(p=self.p+1)
        if not isinstance(other, Order2MatrixSemiring):
            return NotImplemented
        return Order2MatrixSemiring(self.p + other.p, self.phi + other.phi, self.psi + other.psi, self.C + other.C)

    def __radd__(self, other):
        if other == 0:
            return self
        if other == 1:
            return self._replace(p=self.p+1)
        if not isinstance(other, Order2MatrixSemiring):
            return NotImplemented
        raise ValueError('Incorrect semiring addition.')

    def __mul__(self, other):
        if other == 0:
            return self.zero_like()
        if other == 1:
            return self
        if not isinstance(other, Order2MatrixSemiring):
            return NotImplemented
        return Order2MatrixSemiring(self.p * other.p,
                                    self.p * other.phi + other.p * self.phi,
                                    self.p * other.psi + other.p * self.psi,
                                    self.p * other.C + other.p * self.C + np.outer(self.phi, other.psi)
                                    + np.outer(other.phi, self.psi))

    def __rmul__(self, other):
        if other == 0:
            return self.zero_like()
        if other == 1:
            return self
        return NotImplemented

    def __eq__(self, other):
        if other in (0, 1):
            return self.p == other and np.all(self.phi == 0) and np.all(self.psi == 0) and np.all(self.C == 0)
        if not isinstance(other, Order2MatrixSemiring):
            return NotImplemented
        return self.p == other.p and self.D == other.D and np.array_equal(self.phi, other.phi) \
            and np.array_equal(self.psi, other.psi) and np.array_equal(self.C, other.C)

    def __ne__(self, other):  # Overrides tuple __ne__ just in case
        if not isinstance(other, Order2MatrixSemiring):
            return NotImplemented
        return not self == other

    def __hash__(self):
        if self.p in (0, 1) and np.all(self.phi == 0) and np.all(self.psi == 0) and np.all(self.C == 0):
            return self.p
        return super(Order2MatrixSemiring, self).__hash__()


class Order2VectSemiring(namedtuple('Order2VectSemiring', ['p', 'phi', 'psi', 'C'])):
    """
    A class for a 2nd order vectorised semiring
    q = number
    phi = D-dimensional vector
    psi = D-dimensional vector
    C = D-dimensional vector
    """
    __slots__ = ()  # Keeps memory low by stopping instance dictionary being created

    @property
    def D(self):
        return self.phi.shape[0]

    def one_like(self):
        D = self.D
        return Order2VectSemiring(1, np.zeros(D), np.zeros(D), np.zeros(D))

    def zero_like(self):
        D = self.D
        return Order2VectSemiring(0, np.zeros(D), np.zeros(D), np.zeros(D))

    def __add__(self, other):
        if other == 0:
            return self
        if other == 1:
            return self._replace(p=self.p+1)
        if not isinstance(other, Order2VectSemiring):
            return NotImplemented
        return Order2VectSemiring(self.p + other.p, self.phi + other.phi, self.psi + other.psi, self.C + other.C)

    def __radd__(self, other):
        if other == 0:
            return self
        if other == 1:
            return self._replace(p=self.p+1)
        if not isinstance(other, Order2VectSemiring):
            return NotImplemented
        raise ValueError('Incorrect semiring addition.')

    def __mul__(self, other):
        if other == 0:
            return self.zero_like()
        if other == 1:
            return self
        if not isinstance(other, Order2VectSemiring):
            return NotImplemented
        return Order2VectSemiring(self.p * other.p,
                                  self.p * other.phi + other.p * self.phi,
                                  self.p * other.psi + other.p * self.psi,
                                  self.p * other.C + other.p * self.C + self.phi * other.psi + other.phi * self.psi)

    def __rmul__(self, other):
        if other == 0:
            return self.zero_like()
        if other == 1:
            return self
        return NotImplemented

    def __eq__(self, other):
        if other in (0, 1):
            return self.p == other and np.all(self.phi == 0) and np.all(self.psi == 0) and np.all(self.C == 0)
        if not isinstance(other, Order2VectSemiring):
            return NotImplemented
        return self.p == other.p and self.D == other.D and np.array_equal(self.phi, other.phi) \
            and np.array_equal(self.psi, other.psi) and np.array_equal(self.C, other.C)

    def __ne__(self, other):  # Overrides tuple __ne__ just in case
        if not isinstance(other, Order2VectSemiring):
            return NotImplemented
        return not self == other

    def __hash__(self):
        if self.p in (0, 1) and np.all(self.phi == 0) and np.all(self.psi == 0) and np.all(self.C == 0):
            return self.p
        return super(Order2VectSemiring, self).__hash__()
