import pytest

from planner.duty_roster import DutyRoster


def test_duty_roster_calculate_duration_same_day_and_overnight():
    assert DutyRoster.calculate_duration("0900", "1030") == pytest.approx(1.5)
    assert DutyRoster.calculate_duration("2300", "0100") == pytest.approx(2.0)
    assert DutyRoster.calculate_duration(900, 1030) == pytest.approx(1.5)


def test_duty_roster_add_day_and_get_duty_roster():
    roster = DutyRoster()
    roster.add_day("Monday")
    assert roster.get_duty_roster() == {"Monday": {}}


def test_duty_roster_add_duty_populates_all_days():
    roster = DutyRoster()
    roster.add_day("Monday")
    roster.add_day("Tuesday")
    roster.add_duty("Hall Duty", "AM", "0900", "1000", 1, 2)

    monday = roster.get_duty_roster()["Monday"]["Hall Duty"]
    tuesday = roster.get_duty_roster()["Tuesday"]["Hall Duty"]

    assert monday["session"] == "AM"
    assert monday["duration"] == pytest.approx(1.0)
    assert monday["min_requirement"] == 1
    assert monday["ideal_case"] == 2
    assert monday["assignees"] == []
    assert tuesday == monday
