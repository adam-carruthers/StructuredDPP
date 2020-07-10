import weakref


class Node:
    def __init__(self, parent=None, children=None):
        self._parent = weakref.ref(parent) if parent else None
        self.children = {*children} if children else None
        self.generated_messages = {}

    def set_children(self, children):
        self.children = {*children}

    def add_child(self, child):
        self.children.add(child)

    def update_children(self, children):
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

    def get_connected_nodes(self):
        yield from self.children
        if self._parent:
            yield self.parent


class Variable(Node):
    def __init__(self, allowed_values, parent=None, children=None):
        super(Variable, self).__init__(parent, children)
        self.allowed_values = allowed_values


class Factor(Node):
    def __init__(self, get_weight, parent=None, children=None):
        super(Factor, self).__init__(parent, children)
        self.get_weight = get_weight


class FactorTree:
    def __init__(self):
        pass
