from unittest import TestCase
from factor_tree import *


class TestNode(TestCase):
    def test_get_connected_nodes(self):
        parent_node = Node()
        main_node = Node()
        child_node_1, child_node_2 = Node(), Node()
        self.assertEqual(
            list(main_node.get_connected_nodes()),
            [],
            "Unconnected node yielded connected nodes"
        )

        main_node.add_child(child_node_1)
        main_node.add_child(child_node_1)
        main_node.add_children([child_node_2])
        yielded_children = list(main_node.get_connected_nodes())
        self.assertEqual(
            len(yielded_children),
            2,
            "Identical children were yielded more than once."
        )
        self.assertIn(
            child_node_1,
            yielded_children,
            "Child not yielded in connected nodes."
        )
        self.assertIn(
            child_node_2,
            yielded_children,
            "Child not yielded in connected nodes."
        )

        yielded_children_excluding = list(main_node.get_connected_nodes(excluding=child_node_1))
        self.assertListEqual(
            yielded_children_excluding,
            [child_node_2],
            "Yielding with exclusion did not exclude child."
        )

        main_node.parent = parent_node
        yielded_with_parent = list(main_node.get_connected_nodes())
        self.assertEqual(
            len(yielded_with_parent),
            3,
            "Parent not yielded in connected nodes."
        )
        self.assertIn(
            parent_node,
            yielded_with_parent,
            "Parent not yielded in connected nodes."
        )

        yielded_parent_excluding = list(main_node.get_connected_nodes(excluding=parent_node))
        self.assertListEqual(
            yielded_parent_excluding,
            yielded_children,
            "Yielding with exclusion did not exclude parent."
        )

    def test_str(self):
        parent_node = Node(name='Beth')
        main_node = Node(parent=parent_node)
        child_node_1, child_node_2, child_node_3 = Node(), Node(), Node()
        self.assertEqual(
            str(parent_node),
            "Beth(parent=None,0 children)",
            "Name of named unconnected node wrong."
        )
        self.assertEqual(
            str(main_node),
            "Node(parent=Beth,0 children)",
            "Name of child node wrong."
        )
        main_node.add_children([child_node_1, child_node_2])
        main_node.add_child(child_node_3)
        self.assertEqual(
            str(main_node),
            "Node(parent=Beth,3 children)",
            "Name of node with children wrong."
        )


class TestFactor(TestCase):
    def test_get_consistent_assignments(self):
        parent_var = Variable(allowed_values=[0,1,2])
        factor = Factor(None, parent=parent_var)
        child_var_1, child_var_2 = Variable(allowed_values='ab'), Variable(allowed_values='cd')

        self.assertListEqual(
            list(factor.get_consistent_assignments(parent_var, 1)),
            [{parent_var: 1}],
            'Getting consistent assignments failed for a factor connected to a single variable.'
        )

        factor.add_children([child_var_1])
        self.assertListEqual(
            list(factor.get_consistent_assignments(parent_var, 1)),
            [
                {parent_var: 1, child_var_1: 'a'},
                {parent_var: 1, child_var_1: 'b'}
            ],
            'Getting consistent assignments failed for a factor connected to two variables.'
        )

        factor.children = [child_var_1, child_var_2]
        self.assertListEqual(
            list(factor.get_consistent_assignments(child_var_1, 'a')),
            [
                {child_var_1: 'a', child_var_2: 'c', parent_var: 0},
                {child_var_1: 'a', child_var_2: 'c', parent_var: 1},
                {child_var_1: 'a', child_var_2: 'c', parent_var: 2},
                {child_var_1: 'a', child_var_2: 'd', parent_var: 0},
                {child_var_1: 'a', child_var_2: 'd', parent_var: 1},
                {child_var_1: 'a', child_var_2: 'd', parent_var: 2},
            ],
            'Getting consistent assignments failed for a factor connected to three variables.'
        )
