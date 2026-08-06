import pytest

from planner.duty_roster import DutyRoster
from planner.utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_DURATION,
    DUTY_END_TIME,
    DUTY_IDEAL_CASE,
    DUTY_MIN_REQUIREMENT,
    DUTY_REQUIRED_FUNCTION,
    DUTY_RESTRICTED_FUNCTION,
    DUTY_START_TIME,
    DUTY_STAFF_PREFERENCE,
)


def test_duty_roster_calculate_duration_same_day_and_overnight():
    assert DutyRoster.calculate_duration("0900", "1030") == pytest.approx(1.5)
    assert DutyRoster.calculate_duration("2300", "0100") == pytest.approx(2.0)
    assert DutyRoster.calculate_duration(900, 1030) == pytest.approx(1.5)
    assert DutyRoster.calculate_duration(800.0, 830.0) == pytest.approx(0.5)
    assert DutyRoster.calculate_duration("800.0", "830.0") == pytest.approx(0.5)


def test_duty_roster_add_day_and_get_duty_roster():
    roster = DutyRoster()
    roster._add_day("Monday")
    assert roster.get_duty_roster() == {"Monday": {}}


def test_duty_roster_add_duty_populates_requested_day_only():
    roster = DutyRoster()
    roster._add_day("Monday")
    roster._add_day("Tuesday")
    roster.add_duty(
        day="Monday",
        activity="Hall Duty",
        session="AM",
        start_time="0900",
        end_time="1000",
        min_requirement=1,
        ideal_case=2,
        required_function=None,
        restricted_function=None,
        staff_preference="Teacher First"
    )

    monday = roster.get_duty_roster()["Monday"]["Hall Duty"]

    assert monday["session"] == "AM"
    assert monday[DUTY_DURATION] == pytest.approx(1.0)
    assert monday[DUTY_MIN_REQUIREMENT] == 1
    assert monday[DUTY_IDEAL_CASE] == 2
    assert monday[DUTY_ASSIGNEES] == []
    assert "Hall Duty" not in roster.get_duty_roster()["Tuesday"]


def test_duty_roster_add_duty_creates_day_with_staff_attributes():
    roster = DutyRoster()
    roster.add_duty(
        day="Monday",
        activity="Hall Duty",
        session="AM",
        start_time="0900",
        end_time="1000",
        min_requirement=1,
        ideal_case=2,
        required_function="First Aid",
        restricted_function="No Labs",
        staff_preference="Teacher First",
    )

    monday = roster.get_duty_roster()["Monday"]["Hall Duty"]
    assert monday[DUTY_REQUIRED_FUNCTION] == "First Aid"
    assert monday[DUTY_RESTRICTED_FUNCTION] == "No Labs"
    assert monday[DUTY_STAFF_PREFERENCE] == "Teacher First"
    assert monday[DUTY_ASSIGNEES] == []
