"""
The decorators in this script are for factor weight calculating.
They take the assignment dictionary and simplify it.
"""
import functools


def convert_var_assignment(func):
    @functools.wraps(func)
    def new_func(assignment):
        return func(*assignment.values())
    return new_func
