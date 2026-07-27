import numpy as np
import pytest

from planner.person import Person


def test_person_time_to_index_calculates_correct_index():
    assert Person.time_to_index("0900") == 0
    assert Person.time_to_index("0930") == 1
    assert Person.time_to_index("1400") == 10
    assert Person.time_to_index("2359") == 29


def test_person_get_name_returns_name():
    person = Person("Alice")
    assert person.get_name() == "Alice"


def test_person_get_availability_returns_empty_for_missing_day():
    person = Person("Alice")
    assert person.get_availability("Monday") == []


def test_person_set_availability_initializes_and_updates_slots():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "1030", 0)

    availability = person.get_availability("Monday")
    assert len(availability) == 18
    assert availability[:3] == [0, 0, 0]
    assert availability[3:] == [-1] * 15

    person.set_availability("Monday", "1000", "1100", 1)
    assert availability[2] == 1
    assert availability[3] == 1


def test_person_check_availability_true_for_available_range():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "1100", 0)
    assert person.check_availability("Monday", "0930", "1100")


def test_person_check_availability_false_for_missing_day():
    person = Person("Alice")
    assert not person.check_availability("Monday", "0900", "0930")


def test_person_check_availability_false_for_unavailable_range():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "0930", 1)
    assert not person.check_availability("Monday", "0900", "0930")


def test_person_get_work_capacity_ratio_zero_without_availability():
    person = Person("Alice")
    assert person.get_work_capacity_ratio() == 0.0


def test_person_get_work_capacity_ratio_calculates_correctly():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "0930", 1)
    person.set_availability("Monday", "0930", "1100", 0)
    assert person.get_work_capacity_ratio() == pytest.approx(0.25)


def test_person_get_hours_worked_counts_on_duty_slots():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "1000", 1)
    assert person.get_hours_worked() == 1.0


def test_person_get_hours_in_school_counts_available_and_on_duty():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "1000", 1)
    person.set_availability("Monday", "1000", "1100", 0)
    assert person.get_hours_in_school() == 2.0


def test_person_add_duty_appends_day():
    person = Person("Alice")
    person.add_duty("Monday")
    assert person._days_assigned == ["Monday"]
