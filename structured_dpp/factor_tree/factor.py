from functools import reduce
from types import MethodType

from structured_dpp.semiring import MaxProductValue

from .node import Node
from .run_types import SamplingRun, MaxProductRun


class Factor(Node):
    """
    A factor evaluates variables to which it is connected.
    Given the variables taking certain values it can then evaluate the quality and diversity features
    associated with the factor.
    All connected nodes (parent, children) must be Variables
    """
    def __init__(self, get_weight, parent=None, children=None, name=None):
        """
        :param function get_weight:
            A function that takes the factor and a dictionary of variables with assignments and returns the weight.
        """
        super(Factor, self).__init__(parent, children, name=name if name else 'Factor')
        self._get_weight = MethodType(get_weight, self)

    def get_weight(self, assignments, run=None):
        return self._get_weight(assignments)

    @staticmethod
    def get_assignment_combinations(vars, run=None):
        """
        Generator to yield all possible assignment to the variables in vars, based on their allowed_values.
        :param vars: The variable nodes that can be assigned to.
        :yields: A possible assignment combination for each variable.
        """
        if len(vars) == 0:
            raise Exception('Subassignment function must be given at least one variable to assign.')
        if len(vars) == 1:
            if isinstance(run, SamplingRun) and vars[0] in run.fixed_vars:
                yield {vars[0]: run.fixed_vars[vars[0]]}
            else:
                yield from ({vars[0]: value} for value in vars[0].allowed_values)
        else:
            if isinstance(run, SamplingRun) and vars[0] in run.fixed_vars:
                yield from ({vars[0]: run.fixed_vars[vars[0]], **subassignment}
                            for subassignment in Factor.get_assignment_combinations(vars[1:]))
            else:
                for value in vars[0].allowed_values:
                    yield from ({vars[0]: value, **subassignment}
                                for subassignment in Factor.get_assignment_combinations(vars[1:]))

    def get_consistent_assignments(self, var, value, run=None):
        """
        Generator to yield all assignment combinations relating to the factor consistent with var=value
        :yields: A possible assignment combination for each variable.
        """
        other_vars = list(self.get_connected_nodes(exclude=var))
        if len(other_vars) == 0:
            yield {var: value}
            return
        yield from ({var: value, **subassignment}
                    for subassignment in self.get_assignment_combinations(other_vars, run=run))

    def get_incoming_messages_for_assignment(self, assignment, exclude=None, run=None):
        """
        Get the messages associated with a certain assignment dictionary
        :param dict assignment: The assignment in {variable: value_of_assignment} form.
        :param Node exclude: This node's message will be excluded, probs because it is not yet calculated.
        :param run: Which run or calculation the tree is running, in this function, it may hint to a node to use a
        different way of creating the message (e.g: a different get_weight function for a factor)
        :yields: A message for each assignment.
        """
        var, value = None, None
        try:
            for var, value in assignment.items():
                if var != exclude:
                    yield var.get_outgoing_message(to=self, value=value, run=run)
        except KeyError:
            raise KeyError(f"{var} didn't have message to {self} with value {value} on run {run}")

    def create_message(self, to, value, run=None):
        message = None
        for assignment in self.get_consistent_assignments(to, value):
            # Sum together the weight of each assignment
            # taking into account the value of the assignment to preceeding nodes
            # by taking the product of the messages relating to that assignment
            # (one message from each variable being assigned)

            if len(assignment) > 1:  # The factor is higher in the tree than other variables
                # The message of the assignment is the weight for this factor
                # times by the info from the contributing variables
                assignment_value = self.get_weight(assignment, run=run) * reduce(
                    lambda x, y: x*y,
                    self.get_incoming_messages_for_assignment(assignment, exclude=to, run=run)
                )
            else:  # The factor is a leaf and has no nodes further down to consider. Woop! Easy!
                assert to in assignment  # to better be the key or else we're calculating stuff for unconnected nodes
                assignment_value = self.get_weight(assignment, run=run)

            if isinstance(run, MaxProductRun):
                message = message if message is not None and assignment_value < message.v else MaxProductValue(assignment_value, assignment)
            else:
                message = message + assignment_value if message is not None else assignment_value

        return message

    def create_all_messages_to(self, to, run=None):
        if self.outgoing_messages.get(run, None) is None:
            self.outgoing_messages[run] = {}
        new_messages = {
            val: self.create_message(to, val, run=run) for val in to.allowed_values
        }
        self.outgoing_messages[run][to] = new_messages
        return new_messages
