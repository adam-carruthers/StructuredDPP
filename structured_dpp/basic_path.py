from structured_dpp.factor_tree import FactorTree, Factor, Variable
from structured_dpp.semiring import Order2VectSemiring as O2VS
import numpy as np
import scipy.linalg as scila
import scipy.stats as scistat

# First, gotta define some stuff to be able to compute the paths
# This is where we define the functions that calculate the weight
N_POSSIBLE_POSITIONS = 3
POSSIBLE_POSITIONS = np.arange(N_POSSIBLE_POSITIONS)
ZEROS_VECTOR = np.zeros(N_POSSIBLE_POSITIONS)
ZEROS_MATRIX = np.zeros((N_POSSIBLE_POSITIONS, N_POSSIBLE_POSITIONS))


def position_diversity_vector(pos):
    distance = POSSIBLE_POSITIONS - pos
    unnormed = np.exp(-distance**2)
    return unnormed/scila.norm(unnormed)


POSITION_DIVERSITY_VECTORS = {pos: position_diversity_vector(pos) for pos in POSSIBLE_POSITIONS}
POSITION_DIVERSITY_MATRICES = {pos: np.outer(POSITION_DIVERSITY_VECTORS[pos], POSITION_DIVERSITY_VECTORS[pos])
                               for pos in POSSIBLE_POSITIONS}
POSITION_DIVERSITY_O2VS = {
    pos: O2VS(1, POSITION_DIVERSITY_VECTORS[pos], POSITION_DIVERSITY_VECTORS[pos], POSITION_DIVERSITY_MATRICES[pos])
    for pos in POSSIBLE_POSITIONS
}


def one_var_weight(assignments):
    pos = next(iter(assignments.values()))
    return POSITION_DIVERSITY_O2VS[pos]


def root_weight(assignments):
    pos = next(iter(assignments.values()))
    q = pos/50
    pos_dv = POSITION_DIVERSITY_VECTORS[pos]
    return O2VS(q, q*pos_dv, q*pos_dv, q*POSITION_DIVERSITY_MATRICES[pos])


MOVEMENT_SCALE = 5


def transition_weight(assignments):
    pos1, pos2 = assignments.values()
    q = scistat.norm.pdf((pos1 - pos2)/5)
    return O2VS(q, ZEROS_VECTOR, ZEROS_VECTOR, ZEROS_MATRIX)


# Now create the nodes
# Create the start nodes
root = Variable(POSSIBLE_POSITIONS, name='RootVar0')
factor_for_root = Factor(root_weight, parent=root, name='Fac0')
# Then create the rest in a chain
current_var = root
nodes_created = [root, factor_for_root]
for i in range(1, 3):
    transition_factor = Factor(transition_weight, parent=current_var, name=f'Fac{i-1}-{i}')
    current_var = Variable(POSSIBLE_POSITIONS, parent=transition_factor, name=f'Var{i}')
    one_var_factor = Factor(one_var_weight, parent=current_var, name=f'Fac{i}')
    nodes_created.extend((transition_factor, current_var, one_var_factor))

ftree = FactorTree.create_from_connected_nodes(nodes_created)

if __name__ == '__main__':
    # Show the generated Factor Tree
    print(ftree)
    ftree.visualise_graph()

    # Do forward pass
    ftree.run_forward_pass()
    print('Done!')
