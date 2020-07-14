import numpy as np
from collections import namedtuple


class Order2VectSemiring(namedtuple('Order2VectSemiring', ['q', 'phi', 'psi', 'C'])):
    """
    A class for a 2nd order vectorised semiring
    q = number
    psi = D-dimensional vector
    psi = D-dimensional vector
    C = DxD Matrix
    """
    __slots__ = ()  # Keeps memory low by stopping instance dictionary being created

    @property
    def D(self):
        return self.phi.shape[0]

    def one_like(self):
        D = self.D
        return Order2VectSemiring(1, np.zeros(D), np.zeros(D), np.zeros((D, D)))

    def zero_like(self):
        D = self.D
        return Order2VectSemiring(0, np.zeros(D), np.zeros(D), np.zeros((D, D)))

    def __add__(self, other):
        if other == 0:
            return self
        if other == 1:
            return self._replace(q=self.q+1)
        return Order2VectSemiring(self.q + other.q, self.phi + other.phi, self.psi + other.psi, self.C + other.C)

    def __radd__(self, other):
        if other == 0:
            return self
        if other == 1:
            return Order2VectSemiring(self.q + 1, self.phi, self.psi, self.C)
        raise ValueError('Incorrect semiring addition.')

    def __mul__(self, other):
        if other == 0:
            return 0
        if other == 1:
            return self
        return Order2VectSemiring(self.q * other.q,
                                  self.q * other.phi + other.q * self.phi,
                                  self.q * other.psi + other.q * self.psi,
                                  self.q * other.C + other.q * self.C + np.outer(self.phi, other.psi)
                                  + np.outer(other.phi, self.psi))

    def __rmul__(self, other):
        if other == 0:
            return 0
        if other == 1:
            return self
        raise ValueError('Incorrect semiring multiplication.')

    def __eq__(self, other):
        if other in (0, 1):
            return self.q == other and np.all(self.phi == 0) and np.all(self.psi == 0) and np.all(self.C == 0)
        return self.q == other.q and self.D == other.D and np.array_equal(self.phi, other.phi) \
            and np.array_equal(self.psi, other.psi) and np.array_equal(self.C, other.C)

    def __ne__(self, other):  # Overrides tuple __ne__ just in case
        return not self == other
