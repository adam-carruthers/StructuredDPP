from enum import Enum, unique
from collections import namedtuple


@unique
class RunCategories(Enum):
    def __repr__(self):
        return f'Run:{self.name}'

    C = 0


class CRun(namedtuple('CRun', ['category'])):
    __slots__ = ()

    def __new__(cls):
        return tuple.__new__(cls, (RunCategories.C,))
C_RUN = CRun()
