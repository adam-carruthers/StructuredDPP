import numpy as np

from structured_dpp.factor_tree.factor import Factor
from structured_dpp.semiring import Order2VectSemiring


class SDPPFactor(Factor):
    """
    A factor class more specialised to DPPs
    """
    def __init__(self, get_quality, get_diversity, get_diversity_matrix=None, parent=None, children=None, name=None):
        super(SDPPFactor, self).__init__(None, parent=parent, children=children, name=name)
        self.get_quality = get_quality
        self.get_diversity = get_diversity
        self.get_diversity_matrix = get_diversity_matrix

    def default_weight(self, assignments):
        q = self.get_quality(assignments)
        dv = self.get_diversity(assignments)
        if self.get_diversity_matrix:
            dvm = self.get_diversity_matrix(assignments)
        else:
            dvm = np.outer(dv, dv)
        return Order2VectSemiring(q, q*dv, q*dv, q*dvm)

    def get_weight(self, assignments):
        pass