from functools import reduce

from .node import Node
from .run_types import QualityOnlySamplingRun, MaxProductRun
from structured_dpp.semiring import MaxProductValue


class Variable(Node):
    """
    A variable represents one part of one item outputted by an SDPP.
    It can take a discrete number of fixed values.
    All connected nodes must be factors.
    """
    def __init__(self, allowed_values, parent=None, children=None, name=None):
        super(Variable, self).__init__(parent, children, name=name if name else 'Variable')
        self.allowed_values = allowed_values

    def get_incoming_messages_for_value(self, value, exclude=None, run=None):
        """
        Get the messages associated with a certain variable value
        :param value: The assignment in {variable: value_of_assignment} form.
        :param Node exclude: This node's message will be excluded, probs because it is not yet calculated.
        :param run: Which run or calculation the tree is running, so that relevant messages are stored correctly.
        It can be any hashable but you should probably use something from run_types
        :yields: A message for each assignment.
        """
        var = None
        try:
            for var in self.get_connected_nodes(exclude=exclude):
                yield var.get_outgoing_message(to=self, value=value, run=run)
        except KeyError:
            raise KeyError(f"{var} didn't have message to {self} with value {value} on run {run}")

    def create_message(self, to, value, run=None):
        incoming_messages = list(self.get_incoming_messages_for_value(value, exclude=to, run=run))
        if incoming_messages:
            return reduce(
                lambda x,y: x*y,
                incoming_messages
            )
        else:
            return 1

    def create_all_messages_to(self, to, run=None):
        if self.outgoing_messages.get(run, None) is None:
            self.outgoing_messages[run] = {}
        new_messages = {
            val: self.create_message(to, val, run=run) for val in self.allowed_values
        }
        self.outgoing_messages[run][to] = new_messages
        return new_messages

    def create_all_messages_when_set(self, set_value, run, exclude=None):
        # Set the outgoing messages when the variable has been set to a particular value by the run
        # All messages where this variable is not set_value are zero
        # Other messages are as normal, however I increase p to reduce roundoff error
        # But I'm not sure if this is okay...
        # It's equivalent to setting a very high quality for this variable having that particular value I think?
        if isinstance(run, QualityOnlySamplingRun):
            self.outgoing_messages[run] = {
                to: {value: 1 if value == set_value else 0
                     for value in self.allowed_values}
                for to in self.get_connected_nodes()
                if to != exclude
            }
        else:
            self.outgoing_messages[run] = {
                to: {value: self.create_message(to, value, run)
                     if value == set_value else 0
                     for value in self.allowed_values}
                for to in self.get_connected_nodes()
                if to != exclude
            }
        return self.outgoing_messages[run]

    def calculate_belief(self, value, run=None):  # TODO: Don't store beliefs in outgoing_messages[run][None]
        """
        Calculates the product of all incoming messages associated with a value
        :param value: The value that the messages being producted are associated to.
        :param run:
        :return: The value of the belief
        """
        return self.create_and_save_message(None, value, run)

    def calculate_all_beliefs(self, run=None):
        """
        Calculates the product of all incoming messages for each possible value of the variable.
        :param run: Which run or calculation the tree is running, so that relevant messages are stored correctly.
        It can be any hashable but you should probably use something from run_types
        :return: A dictionary, the keys being the possible value of the belief and the values being the belief value.
        """
        self.create_all_messages_to(None, run=run)
        return self.outgoing_messages[run][None]

    def calculate_sum_belief(self, recalculate=False, run=None):
        """
        Returns the sum of all the beliefs of this variable
        :param bool recalculate: Whether to recalculate the beliefs or use previously calculated ones (if available)
        :param run: Which run or calculation the tree is running, so that relevant messages are stored correctly.
        It can be any hashable but you should probably use something from run_types
        :return: The sum of the beliefs
        """
        if recalculate or None not in self.outgoing_messages:
            self.calculate_all_beliefs(run=run)
        return sum(self.outgoing_messages[run][None].values())

    def get_belief(self, value, run=None):
        return self.outgoing_messages[run][None][value]

    def calculate_max_message_assignment(self, run):
        if self.outgoing_messages.get(run, None) is None or self.outgoing_messages[run].get(None, None) is None:
            self.calculate_all_beliefs(run)
        max_m, max_m_assignment = -1, None
        for val in self.allowed_values:
            m = self.outgoing_messages[run][None][val]
            if isinstance(m, MaxProductValue):
                m = m.v
            if m > max_m:
                max_m = m
                max_m_assignment = val
        if max_m == -1:
            raise ValueError('Max message assignment did not find a message bigger than -1!')
        return max_m, max_m_assignment
