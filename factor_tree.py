import weakref
from functools import reduce


class Node:
    """
    A node on a standard bidirectional graph with message passing.
    Allows for circular references without causing memory leakage.
    """
    def __init__(self, parent=None, children=None, name=None):
        self._parent = weakref.ref(parent) if parent else None
        self._children = {*children} if children else set()
        self.outgoing_messages = {}
        self.name = name if name else self.__class__.__name__

    # Children
    # The descendant nodes, stored in a set to avoid children being double counted
    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        self._children = {*children}  # To allow it to be set to a list and to then remove duplicates

    def add_child(self, child):
        self.children.add(child)

    def add_children(self, children):
        self.children.update(children)

    # Parent
    # The parent node cannot be referenced directly as that would be a circular reference
    # (assuming that the parent stores this node as a child)
    # but the weakref library fixes that, but means we need to add this extra code.
    @property
    def parent(self):
        """
        Parent setter.
        To avoid circular references use a weakref to refer to parent from the child.
        """
        if not self._parent:
            return self._parent
        _parent = self._parent()
        if _parent:
            return _parent
        else:
            raise LookupError('Parent was destroyed by garbage collector.')

    @parent.setter
    def parent(self, parent):
        self._parent = weakref.ref(parent)

    # Helpful graph traversing functions
    def get_connected_nodes(self, excluding=None):
        """
        Generator that yields all the connected nodes, not distinguishing between parent and children
        :param Node excluding: If this node is encountered it won't be yeilded.
        """
        for x in self.children:
            if x == excluding:
                continue
            yield x
        if self.parent and self.parent != excluding:
            yield self.parent

    # Message functions
    # Messages are all about communicating theoretical value assignments in variables and calculating their weight.
    # All messages are between edge-connected nodes.
    # All messages are about a variable assignment.
    # A message is a certain value, and a single message corresponds to the node it is from, the node it is to,
    # and what the variable has been assigned to.
    # You can think of a message from a factor to a variable as the factor telling the variable about its weight if it
    # is set to a certain value.
    # You can think of a message from a variable to a factor as the variable telling the factor information about the
    # weight of that assignment from nodes lower down in the tree.
    # This allows nodes higher up in the tree to use information about weights lower in the tree to calculate their
    # own weights.
    def get_outgoing_message(self, to, value):
        """
        A function to get the *already calculated* value message from this node to another node
        :param Node to: The node that is being told the value.
        :param value: The value that the message is about.
        :return: The message that was generated.
        """
        return self.outgoing_messages[to][value]

    def create_message(self, to, value):
        """
        A function that calculates the message to node 'to' about value 'value'
        :param Node to: The node that is being told the value.
        :param value: The value being assigned to the node.
        :return: The message corresponding to the assignment.
        """
        raise NotImplementedError()

    def __str__(self):
        return f'{self.name}(parent={self.parent.name if self._parent else None},' \
               f'{len(self.children)} children)'

    def __repr__(self):
        return str(self)


class Variable(Node):
    """
    A variable represents one part of one item outputted by an SDPP.
    It can take a discrete number of fixed values.
    """
    def __init__(self, allowed_values, parent=None, children=None, name=''):
        super(Variable, self).__init__(parent, children, name='Variable'+name)
        self.allowed_values = allowed_values


class Factor(Node):
    """
    A factor evaluates variables to which it is connected.
    Given the variables taking certain values it can then evaluate the quality and diversity features
    associated with the factor.
    """
    def __init__(self, get_weight, parent=None, children=None, name=''):
        super(Factor, self).__init__(parent, children, name='Factor'+name)
        self.get_weight = get_weight

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
        other_vars = list(self.get_connected_nodes(excluding=var))
        if len(other_vars) == 0:
            yield {var: value}
            return
        yield from ({var: value, **subassignment} for subassignment in self.get_assignment_combinations(other_vars))

    def get_incoming_messages_from_assignment(self, assignment):
        """
        Get the messages associated with a certain assignment dictionary
        :param dict assignment: The assignment in {variable: value_of_assignment} form.
        :yields: A message for each assignment.
        """
        for var, value in assignment.items():
            yield var.get_outgoing_message(to=self, value=value)

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
                    self.get_incoming_messages_from_assignment(assignment)
                )
            else:  # The factor is a leaf and has no nodes further down to consider. Woop! Easy!
                assert assignment[0] == to  # The only assignment
                assignment_value = self.get_weight(assignment)

            message = message + assignment_value if message else assignment_value
