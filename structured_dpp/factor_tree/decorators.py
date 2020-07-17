"""
The decorators in this script are for factor weight calculating.
They take the assignment dictionary and simplify it.
"""
import functools


def assignment_to_var_arguments(func):
    @functools.wraps(func)
    def new_func(factor, assignment):
        return func(*(assignment[var] for var in factor.get_connected_nodes()))
    return new_func
