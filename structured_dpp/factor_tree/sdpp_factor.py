import numpy as np

from structured_dpp.factor_tree.factor import Factor
from structured_dpp.semiring import Order2VectSemiring

from .run_types import CRun


class SDPPFactor(Factor):
    """
    A factor class more specialised to DPPs.
    This means that it only needs to be given quality and diversity
    so that you don't have to specify a complex weight function yourself.
    """
    def __init__(self, get_quality, get_diversity, get_diversity_matrix=None, parent=None, children=None, name=None):
        """
        Creates a SDPPFactor
        :param function get_quality: If given the assignment, work out the quality calculated by the factor.
        :param function get_diversity: If given the assignment, calculate the diversity feature vector.
        :param get_diversity_matrix: (Optional) if given the matrix, return the outer product of the diversity feature
        with itself. This can be used for efficiency gains.
        :param parent: The parent node in the factor tree.
        :param children: The children in the factor tree (not recommended to set).
        :param name: A friendly name that will help you work out which factor is which.
        """
        super(SDPPFactor, self).__init__(lambda: None, parent=parent, children=children, name=name)
        self.get_quality = get_quality
        self.get_diversity = get_diversity
        self.get_diversity_matrix = get_diversity_matrix

    def default_weight(self, assignments):
        q = self.get_quality(assignments)**2
        dv = self.get_diversity(assignments)
        if self.get_diversity_matrix:
            dvm = self.get_diversity_matrix(assignments)
        else:
            dvm = np.outer(dv, dv)
        return Order2VectSemiring(q, q*dv, q*dv, q*dvm)

    def get_weight(self, assignments, run=CRun()):
        if run == CRun():
            return self.default_weight(assignments)
        else:
            raise ValueError('When running an SDPP factor run you must choose a valid run type.')