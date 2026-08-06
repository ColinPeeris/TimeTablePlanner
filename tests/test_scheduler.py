import pandas as pd
import pytest

from planner.duty_roster import DutyRoster
from planner.person import Person
from planner.queue import Queue
from planner.scheduler import Scheduler
from planner.staff_attributes import StaffAttributes
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


def test_scheduler_add_to_queue_adds_staff_members():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    staff_list = [
        ["Monday", "AM", " Alice ", 900, 1200],
        ["Monday", "AM", "Bob", 1000, 1400],
    ]

    scheduler._add_to_queue(queue, staff_list)

    names = [person.get_name() for person in queue.get_list()]
    assert names == ["Alice", "Bob"]

    assert queue.get_list()[0].check_availability("Monday_AM", "0900", "1200")
    assert queue.get_list()[1].check_availability("Monday_AM", "1000", "1400")


def test_scheduler_add_to_queue_normalizes_float_times():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    staff_list = [
        ["Monday", "AM", "Alice", 800.0, 1700.0],
    ]

    scheduler._add_to_queue(queue, staff_list)
    availability = queue.get_list()[0].get_availability("Monday_AM")

    assert len(availability) == 20
    assert availability[:18] == [0] * 18
    assert availability[18:] == [-1, -1]


def test_scheduler_assign_staff_to_duty_uses_teacher_before_temp():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Teacher", "Monday", "0900", "1000", 0)
    temp_queue.add_to_queue("Temp", "Monday", "0900", "1000", 0)

    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: None,
        DUTY_RESTRICTED_FUNCTION: None,
        DUTY_STAFF_PREFERENCE: "Teacher First",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)

    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Teacher"


def test_scheduler_assign_staff_to_duty_raises_when_no_staff_available():
    scheduler = object.__new__(Scheduler)
    teacher_queue = Queue()
    temp_queue = Queue()
    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: None,
        DUTY_RESTRICTED_FUNCTION: None,
        DUTY_STAFF_PREFERENCE: "Teacher First",
    }

    with pytest.raises(ValueError):
        scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)


def test_scheduler_assign_staff_to_duty_uses_temp_when_temp_first():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Teacher", "Monday", "0900", "1000", 0)
    temp_queue.add_to_queue("Temp", "Monday", "0900", "1000", 0)

    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: None,
        DUTY_RESTRICTED_FUNCTION: None,
        DUTY_STAFF_PREFERENCE: "Temp First",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)

    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Temp"


def test_scheduler_assign_staff_to_duty_respects_required_and_restricted_functions():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._staff_attributes.add_required_function("Teacher", "Prefect Duty")
    scheduler._staff_attributes.add_restricted_function("Temp", "Prefect Duty")
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Teacher", "Monday", "0900", "1000", 0)
    temp_queue.add_to_queue("Temp", "Monday", "0900", "1000", 0)

    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: "Prefect Duty",
        DUTY_RESTRICTED_FUNCTION: "Prefect Duty",
        DUTY_STAFF_PREFERENCE: "Teacher First",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)

    assert len(duty_info["assignees"]) == 1
    assert duty_info["assignees"][0].get_name() == "Teacher"


def test_scheduler_assign_staff_to_duty_defaults_to_no_preference_when_missing():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Teacher", "Monday", "0900", "1000", 0)
    temp_queue.add_to_queue("Temp", "Monday", "0900", "1000", 0)

    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: None,
        DUTY_RESTRICTED_FUNCTION: None,
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)

    assert len(duty_info["assignees"]) == 1


def test_order_duties_for_assignment_prioritizes_constrained_duties():
    scheduler = object.__new__(Scheduler)
    duties = {
        "Generic Duty": {
            "required_function": None,
            "restricted_function": None,
            "min_requirement": 1,
            "ideal_case": 1,
            "duration": 1.0,
        },
        "Prefect Duty": {
            "required_function": "Prefect Duty",
            "restricted_function": None,
            "min_requirement": 1,
            "ideal_case": 1,
            "duration": 1.0,
        },
        "Restricted Duty": {
            "required_function": None,
            "restricted_function": "No Duty",
            "min_requirement": 1,
            "ideal_case": 1,
            "duration": 1.0,
        },
    }

    ordered = scheduler._order_duties_for_assignment(duties)
    ordered_names = [name for name, _ in ordered]

    assert ordered_names[0] in {"Prefect Duty", "Restricted Duty"}
    assert ordered_names[-1] == "Generic Duty"


def test_scheduler_get_staff_attributes_from_excel_reads_required_and_restricted(tmp_path):
    file_path = tmp_path / "staff_attributes.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Staff Name": ["Alice", "Bob"],
                "Special Function": ["First Aid", "Supervision"],
            }
        ).to_excel(writer, sheet_name="Special Functions", index=False)
        pd.DataFrame(
            {
                "Staff Name": ["Alice", "Carol"],
                "Restrictions": ["No Labs", "No Duty"],
            }
        ).to_excel(writer, sheet_name="Restrictions", index=False)

    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._get_staff_attributes_from_excel(str(file_path))

    assert scheduler._staff_attributes.has_required_function("Alice", "First Aid")
    assert scheduler._staff_attributes.has_required_function("Bob", "Supervision")
    assert scheduler._staff_attributes.has_restriction("Alice", "No Labs")
    assert scheduler._staff_attributes.has_restriction("Carol", "No Duty")


def test_scheduler_get_staff_availability_reads_excel(tmp_path):
    file_path = tmp_path / "availability.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Monday", "AM", "Alice"],
                ["Tuesday", "PM", "Bob"],
            ],
            columns=["Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Teachers", index=False)

        pd.DataFrame(
            [
                ["Wednesday", "AM", "Carol"],
                ["Thursday", "PM", "Dave"],
            ],
            columns=["Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Temps", index=False)
    result = Scheduler._get_staff_availability(str(file_path))
    assert len(result) == 2
    teachers, temps = result
    assert len(teachers) == 2
    assert len(temps) == 2
    assert teachers[0] == ["Monday", "AM", "Alice"]
    assert teachers[1] == ["Tuesday", "PM", "Bob"]
    assert temps[0] == ["Wednesday", "AM", "Carol"]
    assert temps[1] == ["Thursday", "PM", "Dave"]


def test_scheduler_get_duties_list_from_excel_populates_duty_roster(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Day": ["Monday"],
            "Date": ["2026-08-01"],
            "Activity": ["Hall Duty"],
            "Session": ["AM"],
            "Start Time": ["0900"],
            "End Time": ["1000"],
            "Minimum Requirement": [1],
            "Ideal Case": [2],
            "Required Function": [None],
            "Restricted Function": [None],
            "Staff Preference": ["Teacher First"],
        }
    )
    duties.to_excel(file_path, index=False)

    scheduler = object.__new__(Scheduler)
    scheduler._duty_roster = DutyRoster()
    scheduler._get_duties_list_from_excel(str(file_path))

    duty_roster = scheduler._duty_roster.get_duty_roster()
    assert "Monday_2026-08-01" in duty_roster
    assert "Hall Duty" in duty_roster["Monday_2026-08-01"]
    assert duty_roster["Monday_2026-08-01"]["Hall Duty"][DUTY_DURATION] == pytest.approx(1.0)


def test_scheduler_get_duties_list_from_excel_normalizes_float_times(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Day": ["Monday"],
            "Date": ["2026-08-01"],
            "Activity": ["Hall Duty"],
            "Session": ["AM"],
            "Start Time": [800.0],
            "End Time": [1700.0],
            "Minimum Requirement": [1],
            "Ideal Case": [2],
            "Required Function": [None],
            "Restricted Function": [None],
            "Staff Preference": ["Teacher First"],
        }
    )
    duties.to_excel(file_path, index=False)

    scheduler = object.__new__(Scheduler)
    scheduler._duty_roster = DutyRoster()
    scheduler._get_duties_list_from_excel(str(file_path))

    duty_roster = scheduler._duty_roster.get_duty_roster()
    assert "Monday_2026-08-01" in duty_roster
    assert "Hall Duty" in duty_roster["Monday_2026-08-01"]
    assert duty_roster["Monday_2026-08-01"]["Hall Duty"][DUTY_START_TIME] == "0800"
    assert duty_roster["Monday_2026-08-01"]["Hall Duty"][DUTY_END_TIME] == "1700"
    assert duty_roster["Monday_2026-08-01"]["Hall Duty"][DUTY_DURATION] == pytest.approx(9.0)


def test_scheduler_write_roster_to_excel_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roster = {"Monday": {"Hall Duty": {"assignees": [Person("Alice")], "start_time": "0900", "end_time": "1000"}}}
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Alice", "Monday", "0900", "1000", 1)

    Scheduler._write_roster_to_excel(roster, teacher_queue, temp_queue)
    assert (tmp_path / "teacher_schedule_with_duties.xlsx").exists()
