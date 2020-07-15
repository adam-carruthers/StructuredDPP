from .factor_tree import FactorTree
from .factor import Factor
from .sdpp_factor import SDPPFactor
from .variable import Variable
from .run_types import C_RUN, CRun

import scipy.linalg as scila

import logging


logger = logging.getLogger(__name__)


class SDPPFactorTree(FactorTree):
    def __init__(self, root_node):
        if not isinstance(root_node, Variable):
            raise ValueError('For an SDPPFactorTree your root node has to be a variable node.')
        super(SDPPFactorTree, self).__init__(root_node)
        self.C = None
        self._C_eigendecomp = None

    def add_parent_edges(self, parent, *children):
        if any(isinstance(child, Factor) and not isinstance(child, SDPPFactor) for child in children):
            raise ValueError('Cannot add normal factors to an SDPPFactorTree, they must be SDPPFactors')
        super(SDPPFactorTree, self).add_parent_edges(parent, *children)

    def calculate_C(self, run=C_RUN):
        if not isinstance(run, CRun):
            raise ValueError(f'The run category for calculating C was not Run:C, it was {run}')
        self.run_forward_pass(run=run)
        self.C = self.root.calculate_sum_belief(run=run)[3]
        return self.C

    def calculate_C_eigendecompositon(self, recalculate=False):
        """
        Calculates C's eigendecomposition if it hasn't yet been calculated, saves it and returns it.
        :param bool recalculate: Whether to recalculate the eigendecomposition even if one is already saved.
        :return: eigvals, eigvects
        """
        if self.C is None:
            raise ValueError('C has not been calculated yet!')
        if recalculate or self._C_eigendecomp is None:
            logger.info('Calculating C eigendecomposition')
            self._C_eigendecomp = scila.eigh(self.C)
        return self._C_eigendecomp

    @property
    def C_eigenvalues(self):
        return self.calculate_C_eigendecompositon()[0]

    @property
    def C_eigenvectors(self):
        return self.calculate_C_eigendecompositon()[1]
