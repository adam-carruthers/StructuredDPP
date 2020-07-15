from functools import reduce

from structured_dpp.factor_tree.node import Node


class Variable(Node):
    """
    A variable represents one part of one item outputted by an SDPP.
    It can take a discrete number of fixed values.
    All connected nodes must be factors.
    """
    def __init__(self, allowed_values, parent=None, children=None, name=None):
        super(Variable, self).__init__(parent, children, name=name if name else 'Variable')
        self.allowed_values = allowed_values

    def get_incoming_messages_for_value(self, value, exclude=None):
        """
        Get the messages associated with a certain variable value
        :param value: The assignment in {variable: value_of_assignment} form.
        :param Node exclude: This node's message will be excluded, probs because it is not yet calculated.
        :yields: A message for each assignment.
        """
        var = None
        try:
            for var in self.get_connected_nodes(exclude=exclude):
                yield var.get_outgoing_message(to=self, value=value)
        except KeyError:
            raise KeyError(f"{var} didn't have message to {self} with value {value}")

    def create_message(self, to, value):
        incoming_messages = list(self.get_incoming_messages_for_value(value, exclude=to))
        if incoming_messages:
            return reduce(
                lambda x,y: x*y,
                incoming_messages
            )
        else:
            return 1

    def create_all_messages_to(self, to):
        outgoing_messages_to = self.outgoing_messages.get(to, {})
        outgoing_messages_to.update({
            val: self.create_message(to, val) for val in self.allowed_values
        })
        self.outgoing_messages[to] = outgoing_messages_to
        return outgoing_messages_to

    def calculate_beliefs(self):
        self.create_all_messages_to(None)

    def calculate_sum_belief(self, recalculate=False):
        if recalculate or None not in self.outgoing_messages:
            self.calculate_beliefs()
        return sum(self.outgoing_messages[None].values())
