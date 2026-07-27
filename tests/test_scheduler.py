import pandas as pd
import pytest

from planner.duty_roster import DutyRoster
from planner.person import Person
from planner.queue import Queue
from planner.scheduler import Scheduler


def test_scheduler_add_to_queue_for_slot_adds_staff_members():
    scheduler = object.__new__(Scheduler)
    queue = Queue()
    slot_list = [["Monday", "AM", " Alice ", "Bob", pd.NA]]

    scheduler._add_to_queue_for_slot(queue, slot_list, "0900", "1200")

    names = [person.get_name() for person in queue.get_list()]
    assert names == ["Alice", "Bob"]
    assert queue.get_list()[0].check_availability("Monday_AM", "0900", "1200")


def test_scheduler_assign_staff_to_duty_uses_teacher_before_temp():
    scheduler = object.__new__(Scheduler)
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Teacher", "Monday", "0900", "1000", 0)
    temp_queue.add_to_queue("Temp", "Monday", "0900", "1000", 0)

    duty_info = {"start_time": "0900", "end_time": "1000", "assignees": []}
    scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)

    assert len(duty_info["assignees"]) == 1
    assert duty_info["assignees"][0].get_name() == "Teacher"


def test_scheduler_assign_staff_to_duty_raises_when_no_staff_available():
    scheduler = object.__new__(Scheduler)
    teacher_queue = Queue()
    temp_queue = Queue()
    duty_info = {"start_time": "0900", "end_time": "1000", "assignees": []}

    with pytest.raises(ValueError):
        scheduler._assign_staff_to_duty("Monday", duty_info, teacher_queue, temp_queue, required_count=1, ideal_case=False)


def test_scheduler_get_staff_availability_reads_excel(tmp_path):
    file_path = tmp_path / "availability.xlsx"
    writer = pd.ExcelWriter(file_path, engine="openpyxl")
    pd.DataFrame([["Monday", "AM", "Alice"]], columns=["Day", "Session", "Name"]).to_excel(
        writer, sheet_name="Teachers_AM", index=False
    )
    pd.DataFrame([["Tuesday", "PM", "Bob"]], columns=["Day", "Session", "Name"]).to_excel(
        writer, sheet_name="Teachers_PM", index=False
    )
    pd.DataFrame([["Wednesday", "AM", "Carol"]], columns=["Day", "Session", "Name"]).to_excel(
        writer, sheet_name="Temps_AM", index=False
    )
    pd.DataFrame([["Thursday", "PM", "Dave"]], columns=["Day", "Session", "Name"]).to_excel(
        writer, sheet_name="Temps_PM", index=False
    )
    writer.save()

    result = Scheduler._get_staff_availability(str(file_path))
    assert len(result) == 4
    assert result[0][0][0] == "Monday"
    assert result[1][0][0] == "Tuesday"
    assert result[2][0][0] == "Wednesday"
    assert result[3][0][0] == "Thursday"


def test_scheduler_get_duties_list_from_excel_populates_duty_roster(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Activity": ["Hall Duty"],
            "Session": ["AM"],
            "Start Time": ["0900"],
            "End Time": ["1000"],
            "Minimum Requirement": [1],
            "Ideal Case": [2],
        }
    )
    duties.to_excel(file_path, index=False)

    scheduler = object.__new__(Scheduler)
    scheduler._duty_roster = DutyRoster()
    scheduler._duty_roster.add_day("Monday")
    scheduler._get_duties_list_from_excel(str(file_path))

    duty_roster = scheduler._duty_roster.get_duty_roster()
    assert "Hall Duty" in duty_roster["Monday"]
    assert duty_roster["Monday"]["Hall Duty"]["duration"] == pytest.approx(1.0)


def test_scheduler_write_roster_to_excel_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roster = {"Monday": {"Hall Duty": {"assignees": [Person("Alice")], "start_time": "0900", "end_time": "1000"}}}
    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Alice", "Monday", "0900", "1000", 1)

    Scheduler._write_roster_to_excel(roster, teacher_queue, temp_queue)
    assert (tmp_path / "teacher_schedule_with_duties.xlsx").exists()
