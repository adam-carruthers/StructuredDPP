from warnings import warn
import logging

from structured_dpp.factor_tree.factor import Factor
from structured_dpp.factor_tree.node import Node
from structured_dpp.factor_tree.variable import Variable


logger = logging.getLogger(__name__)


class FactorTree:
    def __init__(self, root_node):
        self.root = root_node
        self.levels = [{root_node}]
        self.item_directory = {root_node: 0}

    def add_parent_edges(self, parent, *children):
        """
        Adds an edge in the graph between the parent and the child
        :param Node parent: The parent, will be higher in the tree than child, must already be in the tree.
        :param Node child: The child, a descendant of parent.
        If the parent is a variable node, the child must be a factor, or the reverse.
        """
        # Integrity checks
        if (isinstance(parent, Variable) and any(not isinstance(child, Factor) for child in children)) or \
            (isinstance(parent, Factor) and any(not isinstance(child, Variable) for child in children)) or \
                not (isinstance(parent, Variable) or isinstance(parent, Factor)):
            raise ValueError('The parent must be a variable or factor, and the children must be the opposite of the '
                             'parent. Aka a variable parent must have factor children and visa versa.')
        if any(child.parent not in (None, parent) for child in children):
            raise ValueError(f'Child has parent attribute set already and it is not correct.')
        if any(child in self.item_directory for child in children):
            raise ValueError('Trying to add node that is already in the tree.')

        # Work out which level the parent is at
        parent_level = self.item_directory.get(parent, None)
        if parent_level is None:
            raise KeyError('Parent must be added to tree before you can add children to it.')
        parent.add_children(children)
        for child in children:
            child.parent = parent
            self.item_directory[child] = parent_level + 1
        if len(self.levels) - 1 <= parent_level:  # Does the next level exist yet?
            self.levels.append({*children})
        else:
            self.levels[parent_level + 1].update(children)

    def generate_up_messages_on_level(self, level):
        node: Node
        for node in self.levels[level]:
            node.create_all_messages_to(node.parent)

    def generate_down_messages_on_level(self, level):
        node: Node
        for node in self.levels[level]:
            for child_node in node.children:
                node.create_all_messages_to(child_node)

    def run_forward_pass(self):
        for level in reversed(range(1, len(self.levels))):
            logger.info(f'Forward pass level {level}')
            self.generate_up_messages_on_level(level)

    def run_backward_pass(self):
        for level in range(len(self.levels)):
            logger.info(f'Backward pass level {level}')
            self.generate_down_messages_on_level(level)

    def nodes_to_add_based_on_parents(self, nodes):
        """
        Generator to get the nodes who's parents are in the tree
        Only used for create_from_connected_nodes
        """
        return filter(lambda x: x.parent in self.item_directory, nodes)

    @classmethod
    def create_from_connected_nodes(cls, nodes):
        """
        Takes nodes already connected in a tree structure and creates a FactorTree of them
        :param list nodes: The nodes which must be already connected through the parent attribute.
        The children attribute should be unset!
        :return: Created FactorTree
        """
        # Integrity check
        if any(len(node.children) > 0 for node in nodes):
            warn('Children attributes are already set for node being added to FactorTree. '
                 'Ideally you should let the children attribute be set by the FactorTree. ', RuntimeWarning)

        # Find the root and make sure it's the only one
        parentless_nodes = [node for node in nodes if node.parent is None]
        if len(parentless_nodes) == 0:
            raise ValueError('Nodes passed to FactorTree must have a root node. '
                             'This node would have no parent set.')
        if len(parentless_nodes) > 1:
            raise ValueError('Invalid tree, every node in the tree excluding the root would have a parent.')
        root = parentless_nodes[0]

        # Create the tree!
        ftree = cls(root_node=root)

        # We see if we're done by keeping track of the nodes remaining, and the ones that need to be added next
        nodes_remaining = {*nodes}
        nodes_remaining.remove(root)
        next_to_add = list(ftree.nodes_to_add_based_on_parents(nodes_remaining))
        while True:
            if len(next_to_add) == 0 or len(nodes_remaining) == 0:
                if len(next_to_add) == 0 and len(nodes_remaining) == 0:
                    return ftree  # Success!
                else:
                    raise ValueError('Not all the nodes in the tree were added. '
                                     'This might be because of a cycle in the graph or other invalid structure. '
                                     'Check the tree is valid and that you have listed all the nodes that need to be '
                                     'added to the tree as an argument.')
            node: Node
            for node in next_to_add:
                ftree.add_parent_edges(node.parent, node)
                nodes_remaining.remove(node)
            next_to_add = list(ftree.nodes_to_add_based_on_parents(nodes_remaining))

    def __str__(self):
        return f'FactorTree({len(self.item_directory)} nodes, {len(self.levels)} levels)'

    def __repr__(self):
        return str(self)

    def convert_to_nx_graph(self):
        import networkx as nx

        G = nx.DiGraph()
        for node in self.item_directory.keys():
            if len(node.children) == 0:
                continue
            G.add_edges_from([(node, child) for child in node.children])

        return G

    def visualise_graph(self):
        logger.info('Starting graph visualisation')
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.convert_to_nx_graph()
        pos = nx.drawing.spring_layout(G.to_undirected(), iterations=150)

        nx.draw_networkx_nodes(
            G, pos, node_shape='o', node_color='c',
            nodelist=[node for node in self.item_directory.keys() if isinstance(node, Variable)]
        )
        nx.draw_networkx_nodes(
            G, pos, node_shape='s', node_color='r',
            nodelist=[node for node in self.item_directory.keys() if isinstance(node, Factor)]
        )
        nx.draw_networkx_labels(
            G, pos,
            labels={node: node.name for node in pos.keys()}
        )
        nx.draw_networkx_edges(G, pos)
        plt.show()
