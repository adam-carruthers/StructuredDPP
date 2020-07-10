import weakref


class Node:
    """
    A simple node on a standard bidirectional graph.
    Allows for circular references without causing memory leakage.
    """
    def __init__(self, parent=None, children=None, name=None):
        self._parent = weakref.ref(parent) if parent else None
        self.children = {*children} if children else set()
        self.generated_messages = {}
        self.name = name if name else self.__class__.__name__

    def set_children(self, children):
        self.children = {*children}

    def add_child(self, child):
        self.children.add(child)

    def add_children(self, children):
        self.children.update(children)

    @property
    def parent(self):
        """
        To avoid circular references use a weakref to refer to parent from the child
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

    def create_message(self, to, value):
        pass

    def get_connected_nodes(self, excluding=None):
        for x in self.children:
            if x == excluding:
                continue
            yield x
        if self.parent and self.parent != excluding:
            yield self.parent

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
    def get_subassignment(vars):
        if len(vars) == 0:
            raise Exception('Subassignment function must be given at least one variable to assign.')
        if len(vars) == 1:
            yield from ({vars[0]: value} for value in vars[0].allowed_values)
        else:
            for value in vars[0].allowed_values:
                yield from ({vars[0]: value, **subassignment} for subassignment in Factor.get_subassignment(vars[1:]))

    def get_consistent_assignments(self, var, value):
        """
        Generator to yield all possible assignments to a connected node
        consistent with node=value
        """
        other_vars = list(self.get_connected_nodes(excluding=var))
        if len(other_vars) == 0:
            yield {var: value}
            return
        yield from ({var: value, **subassignment} for subassignment in self.get_subassignment(other_vars))

    def create_message(self, to, value):
        message = None
        for assignment in self.get_consistent_assignments(to, value):
            incoming_assignment_messages = self.get_incoming_assignment_messages(assignment)
            assignment_value = self.get_weight(assignment)


class FactorTree:
    def __init__(self):
        pass
