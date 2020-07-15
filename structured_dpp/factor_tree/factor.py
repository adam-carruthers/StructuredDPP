from functools import reduce
from typing import Optional

from structured_dpp.factor_tree.node import Node


class Factor(Node):
    """
    A factor evaluates variables to which it is connected.
    Given the variables taking certain values it can then evaluate the quality and diversity features
    associated with the factor.
    All connected nodes (parent, children) must be Variables
    """
    def __init__(self, get_weight, parent=None, children=None, name=None):
        """
        :param function get_weight: A function that takes in a dictionary of variables with assignments and returns the
        weight.
        """
        super(Factor, self).__init__(parent, children, name=name if name else 'Factor')
        self._get_weight = get_weight

    def get_weight(self, assignments):
        return self._get_weight(assignments)

    @staticmethod
    def get_assignment_combinations(vars):
        """
        Generator to yield all possible assignment to the variables in vars, based on their allowed_values.
        :param vars: The variable nodes that can be assigned to.
        :yields: A possible assignment combination for each variable.
        """
        if len(vars) == 0:
            raise Exception('Subassignment function must be given at least one variable to assign.')
        if len(vars) == 1:
            yield from ({vars[0]: value} for value in vars[0].allowed_values)
        else:
            for value in vars[0].allowed_values:
                yield from ({vars[0]: value, **subassignment} for subassignment in Factor.get_assignment_combinations(vars[1:]))

    def get_consistent_assignments(self, var, value):
        """
        Generator to yield all assignment combinations relating to the factor consistent with var=value
        :yields: A possible assignment combination for each variable.
        """
        other_vars = list(self.get_connected_nodes(exclude=var))
        if len(other_vars) == 0:
            yield {var: value}
            return
        yield from ({var: value, **subassignment} for subassignment in self.get_assignment_combinations(other_vars))

    def get_incoming_messages_for_assignment(self, assignment, exclude=None):
        """
        Get the messages associated with a certain assignment dictionary
        :param dict assignment: The assignment in {variable: value_of_assignment} form.
        :param Node exclude: This node's message will be excluded, probs because it is not yet calculated.
        :yields: A message for each assignment.
        """
        var, value = None, None
        try:
            for var, value in assignment.items():
                if var != exclude:
                    yield var.get_outgoing_message(to=self, value=value)
        except KeyError:
            raise KeyError(f"{var} didn't have message to {self} with value {value}")

    def create_message(self, to, value):
        message = None
        for assignment in self.get_consistent_assignments(to, value):
            # Sum together the weight of each assignment
            # taking into account the value of the assignment to preceeding nodes
            # by taking the product of the messages relating to that assignment
            # (one message from each variable being assigned)

            if len(assignment) > 1:  # The factor is higher in the tree than other variables
                # The message of the assignment is the weight for this factor
                # times by the info from the contributing variables
                assignment_value = self.get_weight(assignment) * reduce(
                    lambda x, y: x*y,
                    self.get_incoming_messages_for_assignment(assignment, exclude=to)
                )
            else:  # The factor is a leaf and has no nodes further down to consider. Woop! Easy!
                assert to in assignment  # to better be the key or else we're calculating stuff for unconnected nodes
                assignment_value = self.get_weight(assignment)

            message = message + assignment_value if message else assignment_value

        return message

    def create_all_messages_to(self, to):
        outgoing_messages_to = self.outgoing_messages.get(to, {})
        outgoing_messages_to.update({
            val: self.create_message(to, val) for val in to.allowed_values
        })
        self.outgoing_messages[to] = outgoing_messages_to
        return outgoing_messages_to