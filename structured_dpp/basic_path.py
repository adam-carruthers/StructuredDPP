from structured_dpp.factor_tree import SDPPFactorTree, SDPPFactor, Variable, assignment_to_var_arguments
import numpy as np
import scipy.linalg as scila
import scipy.stats as scistat
import logging

logging.basicConfig(level=logging.INFO)

# Constants
N_POSITIONS = 50
N_VARIABLES = 50
MOVEMENT_SCALE = 1
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
def quality_one(*args):
    return 1

@assignment_to_var_arguments
def one_var_diversity(pos):
    return POSITION_DIVERSITY_VECTORS[pos]

@assignment_to_var_arguments
def one_var_diversity_matrix(pos):
    return POSITION_DIVERSITY_MATRICES[pos]

@assignment_to_var_arguments
def transition_quality(pos1, pos2):
    return scistat.norm.pdf((pos1 - pos2) / MOVEMENT_SCALE)

def zero_diversity(*args):
    return ZEROS_VECTOR

def zero_diversity_matrix(*args):
    return ZEROS_MATRIX

@assignment_to_var_arguments
def root_var_quality(pos):
    return pos/N_POSITIONS

@assignment_to_var_arguments
def final_var_quality(pos):
    return 1 if pos == 45 else 0


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
# i += 1
# transition_factor = SDPPFactor(get_quality=transition_quality,
#                                get_diversity=zero_diversity,
#                                get_diversity_matrix=zero_diversity_matrix,
#                                parent=current_var,
#                                name=f'Fac{i-1}-{i}')
# current_var = Variable(POSSIBLE_POSITIONS, parent=transition_factor, name=f'Var{i}')
# one_var_factor = SDPPFactor(get_quality=final_var_quality,
#                             get_diversity=one_var_diversity,
#                             get_diversity_matrix=one_var_diversity_matrix,
#                             parent=current_var,
#                             name=f'Fac{i}')
# nodes_created.extend((transition_factor, current_var, one_var_factor))

ftree = SDPPFactorTree.create_from_connected_nodes(nodes_created)

if __name__ == '__main__':
    # Draw graph
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(13, 13))
    # ftree.visualise_graph()
    # plt.show()
    # Do forward pass
    # C = ftree.calculate_C()
    assignments = ftree.sample_quality_only(k=5)

    # Plot it!
    x = np.arange(N_VARIABLES)
    for i, assignment in enumerate(assignments):
        y = [assignment[var] for var in ftree.get_variables()]
        plt.plot(x, y, label=f'Item {i}')
    plt.legend()
    plt.title('Quality Selected Paths')
    plt.show()
