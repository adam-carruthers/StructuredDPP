from .factor_tree import FactorTree
from .factor import Factor
from .sdpp_factor import SDPPFactor
from .variable import Variable
from .run_types import CRun, SamplingRun, QualityOnlySamplingRun, BaseFixedVarsRun

from structured_dpp.exact_sampling import dpp_eigvals_selector, k_dpp_eigvals_selector, check_random_state

import numpy as np
import scipy.linalg as scila

import logging


logger = logging.getLogger(__name__)


class SDPPFactorTree(FactorTree):
    def __init__(self, root_node: Variable):
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
        self.C = self.root.calculate_sum_belief(run=run).C
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
        if V_hat_eigvects.shape[1] == 0:
            return {}
        return self.run_sample_from_V_hat(V_hat_eigvects=V_hat_eigvects, run_uid=run_uid, random_state=random_state)

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
            sampler=lambda eigvals, rand_state: k_dpp_eigvals_selector(eigvals, k, random_state=rand_state),
            calc_C_eigdec=calc_C_eigdec, run_uid=run_uid, random_state=random_state)

    def sample_quality_only(self, k, run_uid=None, random_state=None):
        rnd = check_random_state(random_state)

        assigments = []
        run = QualityOnlySamplingRun(run_uid)
        self.run_forward_pass(run)
        for i in range(k):
            logger.info('Starting recursive sampling')
            self.backwards_sample_items(rnd, run, quality_only=True)
            logger.info(f'Selected item {i}')
            assigments.append(run.fixed_vars)
            run.fixed_vars = {}

        return assigments

    def run_sample_from_V_hat(self, V_hat_eigvects, run_uid=None, random_state=None):
        """
        Given a selection of eigenvectors V_hat, create a sample from the SDPP
        :param V_hat_eigvects: The selected eigenvectors V_hat.
        Normally a selection of eigenvectors from C divided by sqrt of their eigenvalue
        :param run_uid:
        :param random_state:
        :return: Sample from the SDPP
        """
        rnd = check_random_state(random_state)

        assignments = []
        for k in range(V_hat_eigvects.shape[1], 0, -1):
            # Do a sampling run which will return one sample from the SDPP
            run = SamplingRun(V_hat_eigvects, (run_uid, k))
            self.run_forward_pass(run)
            logger.info('Starting recursive sampling')
            self.backwards_sample_items(rnd, run)
            logger.info(f'Selected item {k}')
            assignments.append(run.fixed_vars)

            if k == 1:
                # We're done!
                return assignments

            Bi = sum(  # This is the feature vector of our new point
                factor.get_diversity(run.fixed_vars)
                for factor in self.get_factors()
            )

            # Now we need to make V_hat orthogonal to Bi
            # First choose an eigenvector to remove
            # It needs length in the direction of Bi > 0
            Bi_dot = V_hat_eigvects.T @ Bi  # Measure how much the eigenvectors are in the direction of the chosen vect.
            index_to_remove = np.argmax(np.abs(Bi_dot))
            eigvect_to_remove = V_hat_eigvects[:, index_to_remove]
            length_of_removed = Bi_dot[index_to_remove]
            if abs(length_of_removed) < 1e-2:
                raise RuntimeError('Could not find eigenvector to remove after selection made')

            # Remove Bi from the eigenvectors
            V_hat_eigvects = V_hat_eigvects - (Bi_dot[np.newaxis, :]/length_of_removed)*eigvect_to_remove[:, np.newaxis]
            assert np.allclose(V_hat_eigvects[:, index_to_remove], 0), \
                "The removed eigenvector should be about zero after the step above."
            V_hat_eigvects = np.delete(V_hat_eigvects, index_to_remove, 1)

            # Orthonormalise the eigenvectors with respect to a non-standard dot product
            # Use Gram-Schmidt
            new_V_hat = []
            for i, eigvect in enumerate(V_hat_eigvects.T):
                eigvect_to_append = eigvect[:, np.newaxis]  # Eigenvector as a column vector
                for orthonormed_eigvect in new_V_hat:
                    eigvect_to_append -= (eigvect.T @ self.C @ orthonormed_eigvect) * orthonormed_eigvect
                eigvect_to_append /= np.sqrt(eigvect_to_append.T @ self.C @ eigvect_to_append)
                new_V_hat.append(eigvect_to_append)

            V_hat_eigvects = np.array(new_V_hat)[:, :, 0].T
            assert V_hat_eigvects.shape[1] == k - 1

    def backwards_sample_items(self, rnd, run: BaseFixedVarsRun, quality_only=False):
        """
        Sample items root downwards, after a sampling forwards pass has been run
        """
        for i in range(0, len(self.levels), 2):  # Selects every variable level
            for var in self.levels[i]:
                item_select_thresh_prob = rnd.rand()
                item_select_cuml_prob = 0
                item_strengths = {
                    item: belief if quality_only else np.sum(belief.C)
                    for item, belief in var.calculate_all_beliefs(run).items()
                }
                strength_total = sum(item_strengths.values())
                logger.debug(f'Strength total {strength_total}')
                for val, strength in item_strengths.items():
                    item_select_cuml_prob += strength/strength_total
                    if item_select_thresh_prob < item_select_cuml_prob:
                        selected_item = val
                        break
                else:
                    raise RuntimeError(f'The SDPP tried to select an item in variable {var} level {self.item_directory[var]}. '
                                       f'The probability of selecting one item should be one. '
                                       f'However the calculated cumulative probability was {item_select_cuml_prob}.')
                logger.debug(f'Selected {selected_item} for {var} (p={item_select_cuml_prob})')

                run.fixed_vars[var] = selected_item
                var.create_all_messages_when_set(selected_item, run, exclude=var.parent)

                child_factor: SDPPFactor
                for child_factor in var.children:
                    for grandchild_var in child_factor.children:
                        child_factor.create_all_messages_to(grandchild_var, run)
