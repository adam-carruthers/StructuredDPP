import weakref
from functools import reduce
from contextlib import contextmanager
from warnings import warn
from typing import List, Set, Any

# Sometimes the factor tree will need to use different multiplicative and additive systems (semiring)
# Hence here we define a global dictionary containing the additive and multiplicative identities
# The additive identity is zero and satisfies
# x + 0 = x
# x * 0 = 0
# The multiplicative identity is one and satisfies
# x * 1 = 1
# The only things that need to be set here are the zero and one, the different systems of addition and multiplication
# can be defined in the objects that are passed through the network.
DEFAULT_SEMIRING_SETTINGS = {
    'zero': 0,
    'one': 1
}
_semiring_settings = DEFAULT_SEMIRING_SETTINGS.copy()
# To ensure that the semiring settings do not cause bugs, we will use a function to set them different within a context
# but always set them back.
# This uses a python feature called context, all code that uses a particular semiring needs to be called in a with.
@contextmanager
def use_different_semiring(zero, one):
    global _semiring_settings
    try:
        _semiring_settings = {'zero': zero, 'one': one}
        yield _semiring_settings
    finally:
        _semiring_settings = DEFAULT_SEMIRING_SETTINGS.copy()


class Node:
    """
    A node on a standard bidirectional graph with message passing.
    Allows for circular references without causing memory leakage.
    """
    def __init__(self, parent=None, children=None, name=None):
        """
        :param Node parent: Parent node in the tree structure
        :param list children: Child nodes in the tree structure
        :param str name: String name for pretty printing
        """
        if parent or children:
            warn("Avoid setting the parent or children attributes on a node at initialisation, "
                 "let the factor tree set them instead.",
                 RuntimeWarning)
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
    def get_connected_nodes(self, exclude=None):
        """
        Generator that yields all the connected nodes, not distinguishing between parent and children
        :param Node exclude: If this node is encountered it won't be yeilded.
        """
        for x in self.children:
            if x != exclude:
                yield x
        if self.parent and self.parent != exclude:
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
        A function that calculates the message to node 'to' about value 'value'.
        Will require that messages further back in the tree are already calculated.
        :param Node to: The node that is being told the value.
        :param value: The value being assigned to the node.
        :return: The message corresponding to the assignment.
        """
        raise NotImplementedError()

    def create_and_save_message(self, to, value):
        message = self.create_message(to, value)
        if self.outgoing_messages.get(to, None):
            self.outgoing_messages[to][value] = message
        else:
            self.outgoing_messages[to] = {value: message}
        return message

    def create_all_messages_to(self, to):
        """
        Creates all the messages you could send to 'to', saves and returns them.
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
            return _semiring_settings['one']

    def create_all_messages_to(self, to):
        outgoing_messages_to = self.outgoing_messages.get(to, {})
        outgoing_messages_to.update({
            val: self.create_message(to, val) for val in self.allowed_values
        })
        self.outgoing_messages[to] = outgoing_messages_to
        return outgoing_messages_to


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
        assert to is not None, "Somehow this node is trying to send a message to None"
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


class FactorTree:
    def __init__(self, root_node):
        self.levels: List[Set[Node]]
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
        if (isinstance(parent, Variable) and any(not isinstance(child, Factor) for child in children)) or \
            (isinstance(parent, Factor) and any(not isinstance(child, Variable) for child in children)) or \
                not (isinstance(parent, Variable) or isinstance(parent, Factor)):
            raise ValueError('The parent must be a variable or factor, and the children must be the opposite of the '
                             'parent. Aka a variable parent must have factor children and visa versa.')
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
            self.generate_up_messages_on_level(level)

    def run_backward_pass(self):
        for level in range(len(self.levels)):
            self.generate_down_messages_on_level(level)
