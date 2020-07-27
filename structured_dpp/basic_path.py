from structured_dpp.factor_tree import SDPPFactorTree, SDPPFactor, Variable, assignment_to_var_arguments
import numpy as np
import scipy.linalg as scila
import scipy.stats as scistat
import logging

logging.basicConfig(level=logging.INFO)

# Constants
N_POSITIONS = 25
N_VARIABLES = 25
MOVEMENT_SCALE = 1
DIVERSITY_SCALE = 5
POSSIBLE_POSITIONS = np.arange(N_POSITIONS)
ZEROS_VECTOR = np.zeros(N_POSITIONS)
ZEROS_MATRIX = np.zeros((N_POSITIONS, N_POSITIONS))


# Precalculate diversity vectors and matrices for transitions
def position_diversity_vector(pos):
    distance = POSSIBLE_POSITIONS - pos
    unnormed = np.exp(-distance**2/5)
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
    return (pos/N_POSITIONS)**2

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


def plot_path_assignments(assignments, ftree, title, fname=None):
    fig, axs = plt.subplots(1, len(assignments)+3, figsize=(10, 6), constrained_layout=False,
                            gridspec_kw={'width_ratios': [0.5]+[7]+[1]*len(assignments)+[0.5], 'wspace': 0, 'left': 0, 'right': 1})
    fig.suptitle(title)
    axs[0].axis('off')
    axs[-1].axis('off')

    x = np.arange(N_VARIABLES)

    for (i, assignment), col in zip(enumerate(assignments), 'bygrmkc'):
        y = [assignment[var] for var in ftree.get_variables()]
        axs[1].plot(x, y, c=col)
        div = sum(  # This is the feature vector of our new point
            factor.get_diversity(assignment)
            for factor in ftree.get_factors()
        )
        axs[i+2].plot(div, range(len(div)), c=col)
        axs[i+2].xaxis.set_visible(False)
        axs[i+2].yaxis.set_visible(False)

    for i in range(2, len(assignments)+2):
        axs[i].set_ylim(axs[1].get_ylim())

    fig.show()
    if fname is not None:
        fig.savefig(fname)
    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Visualise the factor tree
    # plt.figure(figsize=(15, 15))
    # ftree.visualise_graph()
    # plt.show()

    # Do forward pass
    C = ftree.calculate_C()

    for i in range(1, 5):
        # Sample SDPP
        assignments = ftree.sample_from_kSDPP(k=5)
        plot_path_assignments(assignments, ftree, f"kSDPP Selected Paths {i}", f'plots/SDPP{i}.png')

        # Sample MRF
        assignments_q = ftree.sample_quality_only(k=5)
        plot_path_assignments(assignments_q, ftree, f"Quality Selected Paths {i}", f'plots/Quality{i}.png')
