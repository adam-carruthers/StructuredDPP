import weakref


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
    def get_outgoing_message(self, to, value, run=None):
        """
        A function to get the *already calculated* value message from this node to another node
        :param Node to: The node that is being told the value.
        :param value: The value that the message is about.
        :param run: Which run or calculation the tree is running, so that relevant messages are stored correctly.
        It can be any hashable but you should probably use something from run_types
        :return: The message that was generated.
        """
        return self.outgoing_messages[run][to][value]

    def create_message(self, to, value, run=None):
        """
        A function that calculates the message to node 'to' about value 'value'.
        Will require that messages further back in the tree are already calculated.
        :param Node to: The node that is being told the value.
        :param value: The value being assigned to the node.
        :param run: Which run or calculation the tree is running, in this function, it may hint to a node to use a
        different way of creating the message (e.g: a different get_weight function for a factor)
        :return: The message corresponding to the assignment.
        """
        raise NotImplementedError()

    def create_and_save_message(self, to, value, run=None):
        message = self.create_message(run, to, value)
        if self.outgoing_messages.get(to, None):
            self.outgoing_messages[run][to][value] = message
        else:
            self.outgoing_messages[run][to] = {value: message}
        return message

    def create_all_messages_to(self, to, run=None):
        """
        Creates all the messages you could send to 'to', saves and returns them.
        """
        raise NotImplementedError()

    def __str__(self):
        return f'{self.name}(parent={self.parent.name if self._parent else None},' \
               f'{len(self.children)} children)'

    def __repr__(self):
        return str(self)