from collections import defaultdict


class StaffAttributes:
    """Store and query special function requirements and restrictions for staff."""

    def __init__(self):
        self._required_functions = defaultdict(set)
        self._restricted_functions = defaultdict(set)

    def add_required_function(self, staff_name, function):
        """Record a required special function for a staff member."""
        self._required_functions[staff_name].add(function)

    def add_restricted_function(self, staff_name, restriction):
        """Record a restricted function for a staff member."""
        self._restricted_functions[staff_name].add(restriction)

    def has_required_function(self, staff_name, function):
        """Return True when no required function is needed or the staff member meets the requirement."""
        if not function:
            return True
        return function in self._required_functions.get(staff_name, set())

    def has_restriction(self, staff_name, restriction):
        """Return True when the staff member is restricted from the given function."""
        if not restriction:
            return False
        return restriction in self._restricted_functions.get(staff_name, set())