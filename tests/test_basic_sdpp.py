from unittest.case import TestCase

import numpy as np
import scipy.linalg as scila
import scipy.stats as scistat

from structured_dpp.factor_tree import *

# Constants
n_positions = 3
n_variables = 3
movement_scale = 5
possible_positions = np.arange(n_positions)
possible_paths = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 2],
    [0, 2, 0],
    [0, 2, 1],
    [0, 2, 2],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 2],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
    [2, 0, 0],
    [2, 0, 1],
    [2, 0, 2],
    [2, 1, 0],
    [2, 1, 1],
    [2, 1, 2],
    [2, 2, 0],
    [2, 2, 1],
    [2, 2, 2],
]


# Precalculate diversity vectors and matrices for transitions
def position_diversity_vector(pos):
    distance = possible_positions - pos
    unnormed = np.exp(-distance ** 2)
    return unnormed / scila.norm(unnormed)


position_diversity_vectors = {pos: position_diversity_vector(pos) for pos in possible_positions}
position_diversity_matrices = {pos: np.outer(position_diversity_vectors[pos], position_diversity_vectors[pos])
                               for pos in possible_positions}

# Constants
zeros_vector = np.zeros(n_positions)
zeros_matrix = np.zeros((n_positions, n_positions))


# Specify the quality and diversity factors for the SDPP
def quality_one(*args):
    return 1


@assignment_to_var_arguments
def one_var_diversity(pos):
    return position_diversity_vectors[pos]


@assignment_to_var_arguments
def one_var_diversity_matrix(pos):
    return position_diversity_matrices[pos]


@assignment_to_var_arguments
def transition_quality(pos1, pos2):
    return scistat.norm.pdf((pos1 - pos2) / movement_scale)


def zero_diversity(*args):
    return zeros_vector


def zero_diversity_matrix(*args):
    return zeros_matrix


@assignment_to_var_arguments
def root_var_quality(pos):
    return pos / n_positions


def create_basic_nodes():
    # Create the start nodes
    root = Variable(possible_positions, name='RootVar0')
    factor_for_root = SDPPFactor(get_quality=root_var_quality,
                                 get_diversity=one_var_diversity,
                                 get_diversity_matrix=one_var_diversity_matrix,
                                 parent=root,
                                 name='Fac0')
    # Then create the rest in a chain
    transition_factor0_1 = SDPPFactor(get_quality=transition_quality,
                                      get_diversity=zero_diversity,
                                      get_diversity_matrix=zero_diversity_matrix,
                                      parent=root,
                                      name='Fac0-1')
    var_1 = Variable(possible_positions, parent=transition_factor0_1, name=f'Var1')
    one_var_factor_1 = SDPPFactor(get_quality=quality_one,
                                  get_diversity=one_var_diversity,
                                  get_diversity_matrix=one_var_diversity_matrix,
                                  parent=var_1,
                                  name=f'Fac1')
    # Final part of chain
    transition_factor1_2 = SDPPFactor(get_quality=transition_quality,
                                      get_diversity=zero_diversity,
                                      get_diversity_matrix=zero_diversity_matrix,
                                      parent=var_1,
                                      name='Fac1-2')
    var_2 = Variable(possible_positions, parent=transition_factor1_2, name=f'Var2')
    one_var_factor_2 = SDPPFactor(get_quality=quality_one,
                                  get_diversity=one_var_diversity,
                                  get_diversity_matrix=one_var_diversity_matrix,
                                  parent=var_2,
                                  name=f'Fac1')

    return root, factor_for_root, transition_factor0_1, var_1, one_var_factor_1, transition_factor1_2, \
        var_2, one_var_factor_2


# noinspection DuplicatedCode
class TestBasicSDPP(TestCase):
    def test_basic_example(self):
        # Create the nodes
        root, factor_for_root, transition_factor0_1, var_1, one_var_factor_1, transition_factor1_2, \
        var_2, one_var_factor_2 = create_basic_nodes()

        # Create the factor tree and run the algorithm
        ftree = FactorTree.create_from_connected_nodes([root, factor_for_root,
                                                        transition_factor0_1,
                                                        var_1, one_var_factor_1,
                                                        transition_factor1_2,
                                                        var_2, one_var_factor_2])
        ftree.run_forward_pass(run=C_RUN)
        ftree_C = root.calculate_sum_belief(run=C_RUN)[3]

        ftree2 = SDPPFactorTree.create_from_connected_nodes([root, factor_for_root,
                                                            transition_factor0_1,
                                                            var_1, one_var_factor_1,
                                                            transition_factor1_2,
                                                            var_2, one_var_factor_2])
        self.assertTrue(
            np.allclose(ftree_C, ftree2.calculate_C(run_uid=1)),
            "SDPPFactorTree coming up with different result to ftree"
        )

        # We calculate and SDPP the long way
        B_vectors = []
        for p0, p1, p2 in possible_paths:
            path_q = (p0 / n_positions) \
                     * scistat.norm.pdf((p0 - p1) / movement_scale) \
                     * scistat.norm.pdf((p2 - p1) / movement_scale)
            path_dv = position_diversity_vectors[p0] + position_diversity_vectors[p1] + position_diversity_vectors[p2]
            B_vectors.append(path_q * path_dv)
            self.assertAlmostEqual(
                path_q**2,
                factor_for_root.get_weight({root: p0})[0] * transition_factor0_1.get_weight({root: p0, var_1: p1})[0]
                * transition_factor1_2.get_weight({var_1: p1, var_2: p2})[0],
                msg=f"The transition quality does not match for path ({p0} {p1} {p2})"
            )
            self.assertTrue(
                np.all(
                    path_dv == (factor_for_root.get_diversity({root: p0})
                                + one_var_factor_1.get_diversity({var_1: p1})
                                + one_var_factor_2.get_diversity({var_2: p2}))
                ),
                f"The diversity vector does not match for path ({p0} {p1} {p2})"
            )
        B = np.array(B_vectors).T
        calc_C = B @ B.T

        self.assertTrue(
            np.allclose(calc_C, ftree_C),
            "Calculated C was different in SDPP compared to expected value."
        )

        C00 = sum(
            ((p0 / n_positions) * scistat.norm.pdf((p0 - p1) / movement_scale)
             * scistat.norm.pdf((p2 - p1) / movement_scale))**2
            *
            (position_diversity_vectors[p0][0] + position_diversity_vectors[p1][0]
             + position_diversity_vectors[p2][0])**2
            for p0, p1, p2 in possible_paths
        )
        self.assertAlmostEqual(
            ftree_C[0, 0],
            C00,
            msg="Theoretical and calculated value of C[0, 0] don't match"
        )

    def test_sampling_forward_pass(self):
        ftree = SDPPFactorTree.create_from_connected_nodes(create_basic_nodes())
        ftree.calculate_C(20)

        eigvals, eigvects = ftree.calculate_C_eigendecompositon()
        selected_indices = [1, 2]
        k = len(selected_indices)
        selected_eigvects = eigvects[:, selected_indices]
        selected_eigvects /= np.sqrt(eigvals[np.newaxis, selected_indices])

        run = SamplingRun(selected_eigvects, 99)
        ftree.run_forward_pass(run)
        ftree.root.calculate_all_beliefs(run)

        # Root outgoing messages on the sampling run 99 to None
        probability_sum = 0
        for p0 in [0, 1, 2]:
            theoretical = sum(
                ((p0 / n_positions) * scistat.norm.pdf((p0 - p1) / movement_scale)
                 * scistat.norm.pdf((p2 - p1) / movement_scale))**2
                *
                (selected_eigvects.T @ (
                    position_diversity_vectors[p0] + position_diversity_vectors[p1] + position_diversity_vectors[p2])
                 )**2
                for p1 in [0, 1, 2]
                for p2 in [0, 1, 2]
            )
            calculated = ftree.root.outgoing_messages[run][None][p0]
            probability_sum += np.sum(calculated.C)
            self.assertTrue(
                np.allclose(
                    calculated.C,
                    theoretical
                ),
                "SDPPTree calculated incorrect marginal probabilities"
            )
        self.assertAlmostEqual(
            probability_sum / k,
            1,
            msg="Sum of probabilities from first sampling run was not 1"
        )

    def test_sampling_run(self):
        ftree = SDPPFactorTree.create_from_connected_nodes(create_basic_nodes())
        ftree.calculate_C()
        ftree.sample_from_SDPP()
        ftree.sample_from_kSDPP(k=2)
