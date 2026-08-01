import pytest

from planner.staff_attributes import StaffAttributes


def test_staff_attributes_default_behavior():
    attrs = StaffAttributes()
    assert attrs.has_required_function("Alice", None)
    assert attrs.has_required_function("Alice", "")
    assert not attrs.has_restriction("Alice", None)
    assert not attrs.has_restriction("Alice", "")


def test_staff_attributes_add_and_query_required_function():
    attrs = StaffAttributes()
    attrs.add_required_function("Alice", "First Aid")

    assert attrs.has_required_function("Alice", "First Aid")
    assert not attrs.has_required_function("Alice", "Supervision")


def test_staff_attributes_add_and_query_restricted_function():
    attrs = StaffAttributes()
    attrs.add_restricted_function("Bob", "No Labs")

    assert attrs.has_restriction("Bob", "No Labs")
    assert not attrs.has_restriction("Bob", "No Duty")
