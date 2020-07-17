import numpy as np
from types import MethodType

from structured_dpp.factor_tree.factor import Factor
from structured_dpp.semiring import Order2MatrixSemiring, Order2VectSemiring

from .run_types import C_RUN, CRun, SamplingRun


class SDPPFactor(Factor):
    """
    A factor class more specialised to DPPs.
    This means that it only needs to be given quality and diversity
    so that you don't have to specify a complex weight function yourself.
    """  # TODO: Write prettier/better docs
    def __init__(self, get_quality, get_diversity, get_diversity_matrix=None, parent=None, children=None, name=None):
        """
        Creates a SDPPFactor
        :param function get_quality:
            If given the factor and the assignment, work out the quality calculated by the factor.
        :param function get_diversity:
            If given the factor and the assignment, calculate the diversity feature vector.
        :param get_diversity_matrix:
            (Optional) if given the matrix, return the outer product of the diversity feature with itself.
            This can be used for efficiency gains.
        :param parent:
            The parent node in the factor tree.
        :param children:
            The children in the factor tree (not recommended to set).
        :param name:
            A friendly name that will help you work out which factor is which.
        """
        super(SDPPFactor, self).__init__(lambda: None, parent=parent, children=children, name=name)
        self.get_quality = MethodType(get_quality, self)
        self.get_diversity = MethodType(get_diversity, self)
        self.get_diversity_matrix = MethodType(get_diversity_matrix, self) if get_diversity_matrix else None

    def default_weight(self, assignments):
        p = self.get_quality(assignments)**2  # p = q**2, and that took me too long to realise
        dv = self.get_diversity(assignments)
        if self.get_diversity_matrix:
            dvm = self.get_diversity_matrix(assignments)
        else:
            dvm = np.outer(dv, dv)
        return Order2MatrixSemiring(p, p * dv, p * dv, p * dvm)

    def sampling_weight(self, assignments, run: SamplingRun):
        p = self.get_quality(assignments)**2
        dv = self.get_diversity(assignments)
        phi = run.eigvects.T @ dv
        return Order2VectSemiring(p, p * phi, p * phi, p * phi ** 2)

    def get_weight(self, assignments, run=C_RUN):
        if run is None:
            raise ValueError('When running an SDPP factor run you must choose a valid run type.')
        if isinstance(run, CRun):
            return self.default_weight(assignments)
        elif isinstance(run, SamplingRun):
            return self.sampling_weight(assignments, run)
        else:
            raise ValueError('When running an SDPP factor run you must choose a valid run type.')
