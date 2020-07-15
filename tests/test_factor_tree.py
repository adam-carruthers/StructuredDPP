from unittest import TestCase
from structured_dpp.factor_tree import *


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
            {None: {extra_var1: {0: 0, 1: 1, 2: 0.5}}}  # Corresponding to the weight the factor gives the variable values
        )

        ftree.generate_up_messages_on_level(2)
        for var in child_vars1:
            self.assertDictEqual(
                var.outgoing_messages,
                {None: {factor1: {1: 1, 2: 1}}}
            )
        self.assertDictEqual(
            extra_var1.outgoing_messages,
            {None: {factor1: {0: 0, 1: 1, 2: 0.5}}}
        )

        ftree.generate_up_messages_on_level(1)
        self.assertDictEqual(
            factor1.outgoing_messages,
            {None: {root_var: {
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
            }}}
        )
        self.assertDictEqual(
            factor2.outgoing_messages,
            {None: {
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
            }}
        )

        ftree.generate_down_messages_on_level(0)
        self.assertDictEqual(
            root_var.outgoing_messages,
            {None: {
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
            }}
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

        ftree.run_forward_pass(run=-7)

        self.assertDictEqual(
            factor2.outgoing_messages,
            {-7: {
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
            }}
        )

        # Run a backwards pass
        # I would struggle to work out the backwards pass values manually so I'll assume their correct
        # Below we just test that it doesn't crash and does generate a message on lower levels
        ftree.run_backward_pass(run=-7)
        self.assertIn(
            baby_factor,
            extra_var1.outgoing_messages[-7]
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
