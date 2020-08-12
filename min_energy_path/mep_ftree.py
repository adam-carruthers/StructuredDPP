from functools import reduce

from structured_dpp.factor_tree import Factor, MaxProductRun, Variable
from structured_dpp.semiring import MaxProductValue

from min_energy_path.points_sphere import get_nearby_sphere_indexes


class MEPFactor(Factor):
    """
    A factor node for the *very specific case* where the factor is an intermediate node between two variables
    representing two points on a path, using all of the other stuff in the min_energy_path module.
    """
    def __init__(self, transition_qualities: dict, length_cutoff, n_slices_behind, n_slices_ahead, points_info,
                 parent=None, children=None, name=None):
        super(MEPFactor, self).__init__(lambda *args: None, parent, children, name)
        self.transition_qualities = transition_qualities
        self.length_cutoff = length_cutoff
        self.n_slices_behind = n_slices_behind
        self.n_slices_ahead = n_slices_ahead
        self.points_info = points_info

    def get_weight(self, assignments, run=None):
        # Remember that the parent is closer to the root
        # So the quality from rootwards to leafwards is
        return self.transition_qualities[assignments[self.parent]][assignments[next(iter(self.children))]]

    def create_message(self, to, value_of_to, run=None):
        message = None
        parent = self.parent
        fromm: Variable = next(iter(self.children)) if to == parent else parent

        # Work out what set of points this value can reach
        # Changes depending on whether we're going rootwards or leafwards
        if isinstance(fromm, MEPVariable):
            from_possible_values = get_nearby_sphere_indexes(
                value_of_to, self.length_cutoff, self.points_info,
                # to == self.parent ==> to is rootwards ==> from is leafwards
                # to != self.parent ==> to is leafwards ==> from is rootwards
                min_dir_index=fromm.slice_start,
                max_dir_index=fromm.slice_end
            )
        else:
            from_possible_values = fromm.allowed_values

        for value_of_from in from_possible_values:
            assignment_weight = (
                self.transition_qualities[value_of_to].get(value_of_from, 0)
                if to == self.parent else
                self.transition_qualities[value_of_from].get(value_of_to, 0)
            )

            assignment_value = assignment_weight * fromm.get_outgoing_message(to=self, value=value_of_from, run=run)

            if isinstance(run, MaxProductRun):
                message = message if message is not None and assignment_value < message.v else MaxProductValue(
                    assignment_value, {fromm: value_of_from, to: value_of_to})
            else:
                message = message + assignment_value if message is not None else assignment_value

        return message


class MEPVariable(Variable):
    def __init__(self, allowed_values, slice_start, slice_end, parent=None, children=None, name=None):
        self.slice_start = slice_start
        self.slice_end = slice_end
        super(MEPVariable, self).__init__(allowed_values, parent, children, name)
