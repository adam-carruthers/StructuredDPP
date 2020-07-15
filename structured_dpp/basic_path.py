from structured_dpp.factor_tree import FactorTree, SDPPFactor, Variable, convert_var_assignment
import numpy as np
import scipy.linalg as scila
import scipy.stats as scistat
import logging

logging.basicConfig(level=logging.INFO)

# Constants
N_POSITIONS = 50
N_VARIABLES = 50
MOVEMENT_SCALE = 5
POSSIBLE_POSITIONS = np.arange(N_POSITIONS)
ZEROS_VECTOR = np.zeros(N_POSITIONS)
ZEROS_MATRIX = np.zeros((N_POSITIONS, N_POSITIONS))


# Precalculate diversity vectors and matrices for transitions
def position_diversity_vector(pos):
    distance = POSSIBLE_POSITIONS - pos
    unnormed = np.exp(-distance**2)
    return unnormed/scila.norm(unnormed)
POSITION_DIVERSITY_VECTORS = {pos: position_diversity_vector(pos) for pos in POSSIBLE_POSITIONS}
POSITION_DIVERSITY_MATRICES = {pos: np.outer(POSITION_DIVERSITY_VECTORS[pos], POSITION_DIVERSITY_VECTORS[pos])
                               for pos in POSSIBLE_POSITIONS}


# Specify the quality and diversity factors for the SDPP
def quality_one(assignment):
    return 1

@convert_var_assignment
def one_var_diversity(pos):
    return POSITION_DIVERSITY_VECTORS[pos]

@convert_var_assignment
def one_var_diversity_matrix(pos):
    return POSITION_DIVERSITY_MATRICES[pos]

@convert_var_assignment
def transition_quality(pos1, pos2):
    return scistat.norm.pdf((pos1 - pos2) / MOVEMENT_SCALE)

def zero_diversity(assignment):
    return ZEROS_VECTOR

def zero_diversity_matrix(assignment):
    return ZEROS_MATRIX

@convert_var_assignment
def root_var_quality(pos):
    return pos/N_POSITIONS


# Now create the nodes
# Create the start nodes
root = Variable(POSSIBLE_POSITIONS, name='RootVar0')
factor_for_root = SDPPFactor(get_quality=root_var_quality,
                             get_diversity=one_var_diversity,
                             get_diversity_matrix=one_var_diversity_matrix,
                             parent=root,
                             name='Fac0')
# Then create the rest in a chain
current_var = root
nodes_created = [root, factor_for_root]
for i in range(1, N_VARIABLES):
    transition_factor = SDPPFactor(get_quality=transition_quality,
                                   get_diversity=zero_diversity,
                                   get_diversity_matrix=zero_diversity_matrix,
                                   parent=current_var,
                                   name=f'Fac{i-1}-{i}')
    current_var = Variable(POSSIBLE_POSITIONS, parent=transition_factor, name=f'Var{i}')
    one_var_factor = SDPPFactor(get_quality=quality_one,
                                get_diversity=one_var_diversity,
                                get_diversity_matrix=one_var_diversity_matrix,
                                parent=current_var,
                                name=f'Fac{i}')
    nodes_created.extend((transition_factor, current_var, one_var_factor))

ftree = FactorTree.create_from_connected_nodes(nodes_created)

if __name__ == '__main__':
    # Draw graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 13))
    ftree.visualise_graph()
    # Do forward pass
    ftree.run_forward_pass()
    C = root.calculate_sum_belief()[3]
