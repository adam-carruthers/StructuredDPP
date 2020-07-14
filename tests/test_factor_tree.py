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

        yielded_children_excluding = list(main_node.get_connected_nodes(exclude=child_node_1))
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

        yielded_parent_excluding = list(main_node.get_connected_nodes(exclude=parent_node))
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
        parent_var = Variable(allowed_values=[0, 1, 2])
        factor = Factor(lambda: None, parent=parent_var)
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

    def test_create_message(self):
        def get_weight1(assignment):
            return next(iter(assignment.values())) ** 2

        def get_weight2(assignment):
            return sum(value for value in assignment.values())

        parent = Variable(allowed_values=[0, 1, 2, 3], name='ParentVar')
        childless_factor = Factor(get_weight=get_weight1, parent=parent)
        self.assertEqual(
            childless_factor.create_message(to=parent, value=2),
            4,  # With no children the value of the message is just the value of the single connected variable squared
            'Factor created message had wrong value in the case where it was connected to only 1 parent-variable '
            '(aka does not need incoming messages to generate message).'
        )
        parentless_factor = Factor(get_weight=get_weight1, children=[parent])
        self.assertEqual(
            parentless_factor.create_message(to=parent, value=3),
            9,  # With only one other node connection the message is just the value of the single connected var squared.
            'Factor created message had wrong value in the case where it was connected to only 1 child-variable '
            '(aka does not need incoming messages to generate message).'
        )

        children = [Variable(allowed_values=[0, 1, 2], name='Var' + str(i)) for i in range(2)]
        factor = Factor(get_weight=get_weight2, parent=parent, children=children)
        for child in children:
            child.outgoing_messages = {
                factor: {
                    0: 0,
                    1: 1,
                    2: 2
                }
            }

        self.assertEqual(
            factor.create_message(to=parent, value=0),
            (1 + 1) * 1 * 1 + (1 + 2) * 1 * 2 * 2 + (2 + 2) * 2 * 2,
            'Message was not correct value for message to parent with children.'
        )

        self.assertEqual(
            factor.create_message(to=parent, value=2),
            (2 + 1 + 1) * 1 * 1 + (2 + 1 + 2) * 1 * 2 * 2 + (2 + 2 + 2) * 2 * 2
        )


class TestVariable(TestCase):
    def test_create_message(self):
        parent = Factor(lambda: None)
        childless_var = Variable(allowed_values=[22], parent=parent)
        self.assertEqual(
            childless_var.create_message(parent, 9e10),
            1,
            'Variable connected message had wrong value when connected to only 1 parent-factor '
            '(aka did not need incoming messages to generate new message)'
        )
        self.assertEqual(
            childless_var.create_message(parent, 9e10),
            1,
            'Variable generated message was wrong after change in semiring settings.'
        )

        parentless_var = Variable(allowed_values=[-5])
        parentless_var.add_child(parent)
        self.assertEqual(
            parentless_var.create_message(parent, 9e10),
            1,
            'Variable connected message had wrong value when connected to only 1 child-factor '
            '(aka did not need incoming messages to generate new message)'
        )

        children = [Factor(lambda: None) for _ in range(4)]
        var = Variable(allowed_values='ab', parent=parent, children=children)
        for i, child in enumerate(children):
            child.outgoing_messages = {var: {'a': 5, 'b': i + 1}}
        parent.outgoing_messages = {var: {'b': 100}}

        self.assertEqual(
            var.create_message(parent, 'a'),
            5 ** 4
        )
        self.assertEqual(
            var.create_message(children[0], 'b'),
            2 * 3 * 4 * 100
        )


# noinspection DuplicatedCode
class TestFactorTree(TestCase):
    def test_add_parent_edges(self):
        root_var = Variable(allowed_values=[0, 1], name='VarRoot')
        ftree = FactorTree(root_node=root_var)
        children = [Factor(lambda: None, name=f'Fact{i}') for i in range(4)]
        nonroot_var = Variable(allowed_values=[0], name='VarNonRoot')
        with self.assertRaises(KeyError, msg='Was allowed to add children to a node not in the tree'):
            ftree.add_parent_edges(nonroot_var, children[1])
        with self.assertRaises(ValueError, msg='Add edges allowed a factor to have an edge to a different factor'):
            ftree.add_parent_edges(root_var, nonroot_var)
        ftree.add_parent_edges(root_var, children[0])
        ftree.add_parent_edges(root_var, *children[1:])
        for child in children:
            self.assertEqual(
                child.parent, root_var,
                "Child's parent attribute not correctly set by FactorTree"
            )
            self.assertIn(
                child, root_var.children,
                "Child not added to parent's children attribute by FactorTree"
            )

        ftree.add_parent_edges(children[0], nonroot_var)
        self.assertEqual(
            ftree.levels[2],
            {nonroot_var}
        )

    def test_generate_up_messages_on_level(self):
        def get_weight1(assignment):
            return sum(value for value in assignment.values())

        def get_weight2(assignment):
            return 2*sum(value for value in assignment.values())

        def get_weight_3(assignment):
            v = next(iter(assignment.values()))
            return {
                0: 0,
                1: 1,
                2: 0.5
            }[v]

        # Define the nodes of the factor tree
        root_var = Variable(allowed_values=[0, 1, 2, 3], name='VarRoot')
        factor1 = Factor(get_weight=get_weight1, name='Factor1')
        factor2 = Factor(get_weight=get_weight2, name='Factor2')
        child_vars1 = [Variable(allowed_values=[1, 2], name='Var1-' + str(i)) for i in range(2)]
        extra_var1 = Variable(allowed_values=[0, 1, 2], name='ExtraVar1')
        child_vars2 = [Variable(allowed_values=[1, 2], name='Var2-' + str(i)) for i in range(3)]
        baby_factor = Factor(get_weight=get_weight_3)

        # Set up the factor tree
        ftree = FactorTree(root_var)
        ftree.add_parent_edges(root_var, factor1, factor2)
        ftree.add_parent_edges(factor1, extra_var1, *child_vars1)
        ftree.add_parent_edges(factor2, *child_vars2)
        ftree.add_parent_edges(extra_var1, baby_factor)

        with self.assertRaises(KeyError, msg="Error wasn't raised when calculating higher level messages too soon.") \
                as ctx:
            ftree.generate_up_messages_on_level(2)
        self.assertTrue(
            "didn't have message" in str(ctx.exception),
            "The error message produced for calculating a higher level too soon wasn't the correct error message."
        )

        # Now generate messages one level at a time and check that the outgoing messages are correct on those levels
        # are correct.
        ftree.generate_up_messages_on_level(3)
        self.assertDictEqual(
            baby_factor.outgoing_messages,
            {extra_var1: {0: 0, 1: 1, 2: 0.5}}  # Corresponding to the weight the factor gives the variable values
        )

        ftree.generate_up_messages_on_level(2)
        for var in child_vars1:
            self.assertDictEqual(
                var.outgoing_messages,
                {factor1: {1: 1, 2: 1}}
            )
        self.assertDictEqual(
            extra_var1.outgoing_messages,
            {factor1: {0: 0, 1: 1, 2: 0.5}}
        )

        ftree.generate_up_messages_on_level(1)
        self.assertDictEqual(
            factor1.outgoing_messages,
            {root_var: {
                0: (
                    (0 + 1 + 1 + 1) * 1 * 1 * 1 +
                    (0 + 1 + 1 + 2) * 1 * 1 * 0.5 +
                    (0 + 1 + 2 + 1) * 1 * 1 * 1 +
                    (0 + 1 + 2 + 2) * 1 * 1 * 0.5 +
                    (0 + 2 + 1 + 1) * 1 * 1 * 1 +
                    (0 + 2 + 1 + 2) * 1 * 1 * 0.5 +
                    (0 + 2 + 2 + 1) * 1 * 1 * 1 +
                    (0 + 2 + 2 + 2) * 1 * 1 * 0.5
                ),
                1: (
                    (1 + 1 + 1 + 1) * 1 * 1 * 1 +
                    (1 + 1 + 1 + 2) * 1 * 1 * 0.5 +
                    (1 + 1 + 2 + 1) * 1 * 1 * 1 +
                    (1 + 1 + 2 + 2) * 1 * 1 * 0.5 +
                    (1 + 2 + 1 + 1) * 1 * 1 * 1 +
                    (1 + 2 + 1 + 2) * 1 * 1 * 0.5 +
                    (1 + 2 + 2 + 1) * 1 * 1 * 1 +
                    (1 + 2 + 2 + 2) * 1 * 1 * 0.5
                ),
                2: (
                    (2 + 1 + 1 + 1) * 1 * 1 * 1 +
                    (2 + 1 + 1 + 2) * 1 * 1 * 0.5 +
                    (2 + 1 + 2 + 1) * 1 * 1 * 1 +
                    (2 + 1 + 2 + 2) * 1 * 1 * 0.5 +
                    (2 + 2 + 1 + 1) * 1 * 1 * 1 +
                    (2 + 2 + 1 + 2) * 1 * 1 * 0.5 +
                    (2 + 2 + 2 + 1) * 1 * 1 * 1 +
                    (2 + 2 + 2 + 2) * 1 * 1 * 0.5
                ),
                3: (
                    (3 + 1 + 1 + 1) * 1 * 1 * 1 +
                    (3 + 1 + 1 + 2) * 1 * 1 * 0.5 +
                    (3 + 1 + 2 + 1) * 1 * 1 * 1 +
                    (3 + 1 + 2 + 2) * 1 * 1 * 0.5 +
                    (3 + 2 + 1 + 1) * 1 * 1 * 1 +
                    (3 + 2 + 1 + 2) * 1 * 1 * 0.5 +
                    (3 + 2 + 2 + 1) * 1 * 1 * 1 +
                    (3 + 2 + 2 + 2) * 1 * 1 * 0.5
                )
            }}
        )
        self.assertDictEqual(
            factor2.outgoing_messages,
            {
                root_var: {
                    0: (
                        2 * (0 + 1 + 1 + 1) +
                        2 * (0 + 1 + 1 + 2) * 3 +
                        2 * (0 + 1 + 2 + 2) * 3 +
                        2 * (0 + 2 + 2 + 2)
                    ),
                    1: (
                        2 * (1 + 1 + 1 + 1) +
                        2 * (1 + 1 + 1 + 2) * 3 +
                        2 * (1 + 1 + 2 + 2) * 3 +
                        2 * (1 + 2 + 2 + 2)
                    ),
                    2: (
                        2 * (2 + 1 + 1 + 1) +
                        2 * (2 + 1 + 1 + 2) * 3 +
                        2 * (2 + 1 + 2 + 2) * 3 +
                        2 * (2 + 2 + 2 + 2)
                    ),
                    3: (
                        2 * (3 + 1 + 1 + 1) +
                        2 * (3 + 1 + 1 + 2) * 3 +
                        2 * (3 + 1 + 2 + 2) * 3 +
                        2 * (3 + 2 + 2 + 2)
                    )
                }
            }
        )

        ftree.generate_down_messages_on_level(0)
        self.assertDictEqual(
            root_var.outgoing_messages,
            {
                factor1: {
                    0: 72,
                    1: 88,
                    2: 104,
                    3: 120
                },
                factor2: {
                    0: 26,
                    1: 32,
                    2: 38,
                    3: 44
                }
            }
        )

    def test_run_forward_backward_pass(self):
        def get_weight1(assignment):
            return sum(value for value in assignment.values())

        def get_weight2(assignment):
            return 2*sum(value for value in assignment.values())

        def get_weight_3(assignment):
            v = next(iter(assignment.values()))
            return {
                0: 0,
                1: 1,
                2: 0.5
            }[v]

        # Define the nodes of the factor tree
        root_var = Variable(allowed_values=[0, 1, 2, 3], name='VarRoot')
        factor1 = Factor(get_weight=get_weight1, name='Factor1')
        factor2 = Factor(get_weight=get_weight2, name='Factor2')
        child_vars1 = [Variable(allowed_values=[1, 2], name='Var1-' + str(i)) for i in range(2)]
        extra_var1 = Variable(allowed_values=[0, 1, 2], name='ExtraVar1')
        child_vars2 = [Variable(allowed_values=[1, 2], name='Var2-' + str(i)) for i in range(3)]
        baby_factor = Factor(get_weight=get_weight_3)

        # Set up the factor tree
        ftree = FactorTree(root_var)
        ftree.add_parent_edges(root_var, factor1, factor2)
        ftree.add_parent_edges(factor1, extra_var1, *child_vars1)
        ftree.add_parent_edges(factor2, *child_vars2)
        ftree.add_parent_edges(extra_var1, baby_factor)

        ftree.run_forward_pass()

        self.assertDictEqual(
            factor2.outgoing_messages,
            {
                root_var: {
                    0: (
                            2 * (0 + 1 + 1 + 1) +
                            2 * (0 + 1 + 1 + 2) * 3 +
                            2 * (0 + 1 + 2 + 2) * 3 +
                            2 * (0 + 2 + 2 + 2)
                    ),
                    1: (
                            2 * (1 + 1 + 1 + 1) +
                            2 * (1 + 1 + 1 + 2) * 3 +
                            2 * (1 + 1 + 2 + 2) * 3 +
                            2 * (1 + 2 + 2 + 2)
                    ),
                    2: (
                            2 * (2 + 1 + 1 + 1) +
                            2 * (2 + 1 + 1 + 2) * 3 +
                            2 * (2 + 1 + 2 + 2) * 3 +
                            2 * (2 + 2 + 2 + 2)
                    ),
                    3: (
                            2 * (3 + 1 + 1 + 1) +
                            2 * (3 + 1 + 1 + 2) * 3 +
                            2 * (3 + 1 + 2 + 2) * 3 +
                            2 * (3 + 2 + 2 + 2)
                    )
                }
            }
        )

        # Run a backwards pass
        # I would struggle to work out the backwards pass values manually so I'll assume their correct
        # Below we just test that it doesn't crash and does generate a message on lower levels
        ftree.run_backward_pass()
        self.assertIn(
            baby_factor,
            extra_var1.outgoing_messages
        )
        print(extra_var1.outgoing_messages)

    def test_create_from_connected_nodes(self):
        # Tree nodes
        root = Variable([1], name='Root')

        f1_1 = Factor(lambda: None, name='F1_1')  # Not setting the parent yet for extra testing
        v2_1 = Variable([1], parent=f1_1, name='V2_1')
        f3_1 = Factor(lambda: None, parent=v2_1, name='F3_1')
        v4_1 = Variable([1], parent=f3_1, name='V4_1')

        f1_2 = Factor(lambda: None, parent=root, name='F2_1')
        v2_2 = Variable([1], parent=f1_2, name='V2_2')
        v2_3 = Variable([1], parent=f1_2, name='V2_3')

        # Unconnected tree
        with self.assertRaises(ValueError, msg='A graph where there was no clear root node was allowed'):
            FactorTree.create_from_connected_nodes([root, f1_1, v2_1, f3_1, v4_1])

        # Loopy graphs
        f1_1.parent = v4_1
        with self.assertRaises(ValueError, msg='A graph with loops was allowed to be added as a tree'):
            FactorTree.create_from_connected_nodes([root, f1_1, v2_1, f3_1, v4_1])
        with self.assertRaises(ValueError, msg='A graph with loops was allowed to be added as a tree'):
            FactorTree.create_from_connected_nodes([root, f1_1, v2_1, f3_1, v4_1, f1_2, v2_2, v2_3])

        # Correct graphs
        f1_1.parent = root
        self.assertListEqual(
            FactorTree.create_from_connected_nodes([root, f1_1, v2_1, f3_1, v4_1]).levels,
            [{root}, {f1_1}, {v2_1}, {f3_1}, {v4_1}]
        )
        self.assertListEqual(
            FactorTree.create_from_connected_nodes([root, f1_1, v2_1, f3_1, v4_1, f1_2, v2_2, v2_3]).levels,
            [{root}, {f1_1, f1_2}, {v2_1, v2_2, v2_3}, {f3_1}, {v4_1}]
        )
