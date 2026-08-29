import pytest

from planner.person import Person


def test_person_time_to_index_calculates_correct_index():
    # Schedule starts at 07:00 with 30-minute slots.
    assert Person.time_to_index("0700") == 0
    assert Person.time_to_index("0730") == 1
    assert Person.time_to_index("0800") == 2
    assert Person.time_to_index("0900") == 4
    assert Person.time_to_index("1400") == 14
    assert Person.time_to_index("1900") == 24


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

    # 07:00-19:00 = 12 hours = 24 half-hour slots
    assert len(availability) == 24

    # 07:00-09:00 remains unavailable
    assert availability[:4] == [-1, -1, -1, -1]

    # 09:00-10:30 is available
    assert availability[4:7] == [0, 0, 0]

    # 10:30-19:00 remains unavailable
    assert availability[7:] == [-1] * 17

    person.set_availability("Monday", "1000", "1100", 1)

    assert availability[6] == 1
    assert availability[7] == 1


def test_person_set_availability_expands_schedule_for_early_start():
    person = Person("Alice")
    person.set_availability("Monday", "0600", "1700", 0)

    availability = person.get_availability("Monday")

    # Original configured grid is 07:00-19:00 = 24 slots.
    # 06:00-07:00 expands the grid by 2 slots.
    assert len(availability) == 26

    # 06:00-17:00 is available.
    assert availability[:22] == [0] * 22

    # 17:00-19:00 remains unavailable.
    assert availability[22:] == [-1] * 4


def test_person_check_availability_true_for_available_range():
    person = Person("Alice")
    person.set_availability("Monday", "0900", "1100", 0)

    assert person.check_availability(
        "Monday",
        "0930",
        "1100",
    )


def test_person_check_availability_false_for_missing_day():
    person = Person("Alice")

    assert not person.check_availability(
        "Monday",
        "0900",
        "0930",
    )


def test_person_check_availability_false_for_unavailable_range():
    person = Person("Alice")
    person.set_availability(
        "Monday",
        "0900",
        "0930",
        1,
    )

    assert not person.check_availability(
        "Monday",
        "0900",
        "0930",
    )


def test_person_get_work_capacity_ratio_zero_without_availability():
    person = Person("Alice")

    assert person.get_work_capacity_ratio() == 0.0


def test_person_get_work_capacity_ratio_calculates_correctly():
    person = Person("Alice")

    person.set_availability(
        "Monday",
        "0900",
        "0930",
        1,
    )

    person.set_availability(
        "Monday",
        "0930",
        "1100",
        0,
    )

    assert person.get_work_capacity_ratio() == pytest.approx(0.25)


def test_person_get_hours_worked_counts_on_duty_slots():
    person = Person("Alice")

    person.set_availability(
        "Monday",
        "0900",
        "1000",
        1,
    )

    assert person.get_hours_worked() == 1.0


def test_person_get_hours_in_school_counts_available_and_on_duty():
    person = Person("Alice")

    person.set_availability(
        "Monday",
        "0900",
        "1000",
        1,
    )

    person.set_availability(
        "Monday",
        "1000",
        "1100",
        0,
    )

    assert person.get_hours_in_school() == 2.0


def test_person_add_duty_appends_day():
    person = Person("Alice")

    person.add_duty("Monday")

    assert person._days_assigned == ["Monday"]


def test_person_expected_capacity_defaults_to_one():
    person = Person("Alice")

    assert person.get_expected_capacity() == 1.0


def test_person_set_expected_capacity_updates_value():
    person = Person("Alice")
    person.set_expected_capacity(0.5)

    assert person.get_expected_capacity() == 0.5


def test_person_expected_capacity_rejects_non_positive_values():
    with pytest.raises(ValueError):
        Person("Alice", expected_capacity=0)

    person = Person("Alice")
    with pytest.raises(ValueError):
        person.set_expected_capacity(-1)


def test_person_get_work_capacity_ratio_scales_by_expected_capacity():
    person = Person("Alice", expected_capacity=0.5)

    person.set_availability("Monday", "0900", "0930", 1)
    person.set_availability("Monday", "0930", "1100", 0)

    # Fill ratio is 0.25; expected capacity 0.5 doubles the work-to-capacity ratio.
    assert person.get_work_capacity_ratio() == pytest.approx(0.5)


def test_person_expected_capacity_is_stored_per_day():
    person = Person("Ferninda")
    person.set_availability("Monday", "0900", "1100", 0)
    person.set_availability("Tuesday", "0900", "1100", 0)
    person.set_expected_capacity(0.7, day="Monday")
    person.set_expected_capacity(0.5, day="Tuesday")

    assert person.get_expected_capacity("Monday") == 0.7
    assert person.get_expected_capacity("Tuesday") == 0.5
    assert person.get_expected_capacity() == pytest.approx(0.6)


def test_person_get_work_capacity_ratio_uses_the_requested_day():
    person = Person("Ferninda")
    person.set_availability("Monday", "0900", "0930", 1)
    person.set_availability("Monday", "0930", "1000", 0)
    person.set_availability("Tuesday", "0900", "0930", 1)
    person.set_availability("Tuesday", "0930", "1000", 0)
    person.set_expected_capacity(1.0, day="Monday")
    person.set_expected_capacity(0.5, day="Tuesday")

    assert person.get_work_capacity_ratio("Monday") == pytest.approx(0.5)
    assert person.get_work_capacity_ratio("Tuesday") == pytest.approx(1.0)
    assert person.get_work_capacity_ratio() == pytest.approx(2 / 3)