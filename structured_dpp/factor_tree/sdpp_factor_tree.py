from .factor_tree import FactorTree
from .factor import Factor
from .sdpp_factor import SDPPFactor
from .variable import Variable
from .run_types import C_RUN, CRun, SamplingRun

from structured_dpp.exact_sampling import dpp_eigvals_selector, k_dpp_eigvals_selector

import numpy as np
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

    def calculate_C(self, run_uid=None):
        run = CRun(run_uid)
        self.run_forward_pass(run=run)
        self.C = self.root.calculate_sum_belief(run=run)[3]
        return self.C

    def calculate_C_eigendecompositon(self, recalculate=False, err=False):
        """
        Calculates C's eigendecomposition if it hasn't yet been calculated, saves it and returns it.
        :param bool recalculate: Whether to recalculate the eigendecomposition even if one is already saved.
        :param bool err: Whether to throw an error if it has to calculate
        :return: eigvals, eigvects
        """
        if self.C is None:
            raise ValueError('C has not been calculated yet! You can calculate it with SDPPFactorTree.calculate_C()')
        if recalculate or self._C_eigendecomp is None:
            if err:
                raise ValueError("C's eigendecomposition hasn't been calculated yet, "
                                 "you can run it with SDPPFactorTree.calculate_C_eigendecomposition()")
            logger.info('Calculating C eigendecomposition')
            self._C_eigendecomp = scila.eigh(self.C)
        return self._C_eigendecomp

    @property
    def C_eigenvalues(self):
        return self.calculate_C_eigendecompositon(err=True)[0]

    @property
    def C_eigenvectors(self):
        return self.calculate_C_eigendecompositon(err=True)[1]

    def sample_eigenvectors_using_sampler(self, sampler, calc_C_eigdec=True, run_uid=None, random_state=None):
        """
        Sample the eigenvectors using the sampler given
        :param sampler: A function that takes the eigenvalues of C and then the random state as an argument
        :param bool calc_C_eigdec: Whether to calculate C's eigendecomposition if it isn't computed already.
        This can be an expensive step.
        :param run_uid: The UID to associate with the run in the factors.
        :param random_state: The random state to use to set the sampling.
        """
        eigvals, eigvects = self.calculate_C_eigendecompositon(err=not calc_C_eigdec)
        selected_indices = sampler(eigvals, random_state)
        V_hat_eigvects = eigvects[:, selected_indices] / np.sqrt(eigvals[np.newaxis, selected_indices])
        logger.info(f'Selected {V_hat_eigvects.shape[1]} eigenvectors')
        self.run_sample_from_V_hat(V_hat_eigvects=V_hat_eigvects, run_uid=run_uid, random_state=random_state)

    def sample_from_SDPP(self, calc_C_eigdec=True, run_uid=None, random_state=None):
        """
        Create a sample from the SDPP
        :param bool calc_C_eigdec: Whether to calculate C's eigendecomposition if it isn't computed already.
        This can be an expensive step.
        :param run_uid: The UID to associate with the run in the factors.
        :param random_state: The random state to use to set the sampling.
        :return:
        """
        return self.sample_eigenvectors_using_sampler(sampler=dpp_eigvals_selector, calc_C_eigdec=calc_C_eigdec,
                                                      run_uid=run_uid, random_state=random_state)

    def sample_from_kSDPP(self, k, calc_C_eigdec=True, run_uid=None, random_state=None):
        """
        Create a sample from the kSDPP
        :param int k: How many items to sample from the DPP
        :param bool calc_C_eigdec: Whether to calculate C's eigendecomposition if it isn't computed already.
        This can be an expensive step.
        :param run_uid: The UID to associate with the run in the factors.
        :param random_state: The random state to use to set the sampling.
        :return:
        """
        return self.sample_eigenvectors_using_sampler(
            sampler=lambda eigvals, k: k_dpp_eigvals_selector(eigvals, k, random_state=random_state),
            calc_C_eigdec=calc_C_eigdec, run_uid=run_uid, random_state=random_state)

    def run_sample_from_V_hat(self, V_hat_eigvects, run_uid=None, random_state=None):
        """
        Given a selection of eigenvectors V_hat, create a sample from the SDPP
        :param V_hat_eigvects: The selected eigenvectors V_hat.
        Normally a selection of eigenvectors from C divided by sqrt of their eigenvalue
        :param run_uid:
        :param random_state:
        :return: Sample from the SDPP
        """
        run = SamplingRun(V_hat_eigvects, run_uid)
        self.run_forward_pass(run)
        self.root.calculate_all_beliefs(run)
