from enum import Enum, unique
from collections import namedtuple


class BaseRun:
    def __init__(self, uid=None):
        self.uid = uid

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.uid == other.uid

    def __hash__(self):
        return hash((type(self), self.uid))

    def __repr__(self):
        return f'{type(self).__name__}:{self.uid}'


class CRun(BaseRun):
    pass
C_RUN = CRun()


class BaseFixedVarsRun(BaseRun):
    def __init__(self, uid=None):
        super(BaseFixedVarsRun, self).__init__(uid)
        self.fixed_vars = {}


class SamplingRun(BaseFixedVarsRun):
    def __init__(self, eigvects, uid=None):
        super(SamplingRun, self).__init__(uid)
        self.eigvects = eigvects


class QualityOnlySamplingRun(BaseFixedVarsRun):
    def __init__(self, uid=None):
        super(QualityOnlySamplingRun, self).__init__(uid)


class MaxProductRun(BaseRun):
    pass
