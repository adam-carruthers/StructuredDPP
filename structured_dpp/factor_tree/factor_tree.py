from warnings import warn
import logging

from .factor import Factor
from .node import Node
from .variable import Variable
from .run_types import MaxProductRun


logger = logging.getLogger(__name__)


class FactorTree:
    def __init__(self, root_node):
        if not isinstance(root_node, Variable):
            warn('For a factor tree your root node should probably be a variable.')
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

    def get_nodes(self):
        """Iterates through the nodes in the FactorTree"""
        yield from self.item_directory.keys()

    def get_variables(self):
        """Iterates through the variables in the FactorTree"""
        for node in self.get_nodes():
            if isinstance(node, Variable):
                yield node

    def get_factors(self):
        """Iterates through the factors in the FactorTree"""
        for node in self.get_nodes():
            if isinstance(node, Factor):
                yield node

    def generate_up_messages_on_level(self, level, run=None):
        node: Node
        for node in self.levels[level]:
            node.create_all_messages_to(node.parent, run=run)

    def generate_down_messages_on_level(self, level, run=None):
        node: Node
        for node in self.levels[level]:
            for child_node in node.children:
                node.create_all_messages_to(child_node, run=run)

    def run_forward_pass(self, run=None):
        logger.info(f'Starting forward pass on run {run}')
        for level in reversed(range(1, len(self.levels))):
            logger.debug(f'Forward pass level {level} on run {run}')
            self.generate_up_messages_on_level(level, run=run)

    def run_backward_pass(self, run=None):
        logger.info(f'Starting backward pass on run {run}')
        for level in range(len(self.levels)):
            logger.debug(f'Backward pass level {level} on run {run}')
            self.generate_down_messages_on_level(level, run=run)

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
        :param nodes: The nodes which must be already connected through the parent attribute.
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
        for node in self.get_nodes():
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

    def generate_depth_first_traversal(self, start_node: Node):
        traversal = [(start_node, None)]  # TODO: Test this function
        level_above = [(start_node, None)]
        while True:
            current_level = []
            for parent_node, exclude in level_above:
                current_level.extend((child_node, parent_node)
                                     for child_node in parent_node.get_connected_nodes(exclude))
            traversal.extend(current_level)
            level_above = current_level
            if len(current_level) == 0:
                return traversal

    def run_forward_pass_from_traversal(self, traversal, run=None):
        logger.info(f'Starting forward pass on run {run}')
        node: Node
        for node, node_above in reversed(traversal[1:]):
            node.create_all_messages_to(node_above, run)

    def run_backward_pass_from_traversal(self, traversal, run=None):
        logger.info(f'Starting backward pass on run {run}')
        node: Node
        node_above: Node
        for node, node_above in traversal[1:]:
            node_above.create_all_messages_to(node, run)

    def run_max_quality_forward(self, start_node=None, run_uid=None):
        start_node = start_node if start_node else self.root
        if not isinstance(start_node, Variable):
            raise ValueError('Max quality_function runs must start from a Variable')

        run = MaxProductRun(run_uid)
        traversal = self.generate_depth_first_traversal(start_node=start_node)

        self.run_forward_pass_from_traversal(traversal, run)
        start_node.calculate_all_beliefs(run)
        return traversal, run

    def get_max_from_start_assignment(self, start_node, start_assignment, traversal, run):
        assignments = {start_node: start_assignment}
        for node, node_above in traversal[1:]:  # Selects factor levels only
            if isinstance(node, Factor):
                message = node.get_outgoing_message(node_above, assignments[node_above], run)
                assignments.update(message.assignment)

        return assignments

    def get_max_quality(self, start_node=None, run_uid=None):
        start_node = start_node if start_node is not None else self.root
        traversal, run = self.run_max_quality_forward(start_node, run_uid)
        root_max_m, root_max_m_assignment = start_node.calculate_max_message_assignment(run)
        logger.info(f'Max path has quality {root_max_m}, starting assigning')
        return self.get_max_from_start_assignment(start_node, root_max_m_assignment, traversal, run)
