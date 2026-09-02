import pandas as pd
import copy
import pytest

from planner.duty_roster import DutyRoster
from planner.person import Person
from planner.queue import Queue
from planner.scheduler import Scheduler
from planner.staff_attributes import StaffAttributes
from planner.utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_ACTIVITY,
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

    assert len(availability) == 24
    assert availability[2:20] == [0] * 18
    assert availability[:2] + availability[20:] == [-1] * 6


def test_scheduler_add_to_queue_normalizes_excel_date_keys():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    scheduler._add_to_queue(queue, [
        ["Tuesday", pd.Timestamp("2026-09-15"), "Alice", 1100, 1200],
    ])

    assert queue.get_list()[0].check_availability(
        "Tuesday_2026-09-15", "1100", "1200"
    )


def test_scheduler_assign_staff_to_duty_uses_teacher_before_temp():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
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
        DUTY_STAFF_PREFERENCE: "Teachers",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": teacher_queue, "Temps": temp_queue}, required_count=1, ideal_case=False)

    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Teacher"


def test_scheduler_assign_staff_to_duty_raises_when_no_staff_available():
    scheduler = object.__new__(Scheduler)
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
    teacher_queue = Queue()
    temp_queue = Queue()
    duty_info = {
        DUTY_START_TIME: "0900",
        DUTY_END_TIME: "1000",
        DUTY_ASSIGNEES: [],
        DUTY_REQUIRED_FUNCTION: None,
        DUTY_RESTRICTED_FUNCTION: None,
        DUTY_STAFF_PREFERENCE: "Teachers",
    }

    result = scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": teacher_queue, "Temps": temp_queue}, required_count=1, ideal_case=False)
    assert isinstance(result, str)


def test_scheduler_assign_staff_to_duty_uses_temp_when_temp_first():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
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
        DUTY_STAFF_PREFERENCE: "Temps",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": teacher_queue, "Temps": temp_queue}, required_count=1, ideal_case=False)

    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Temp"


def test_scheduler_assign_staff_to_duty_respects_required_and_restricted_functions():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
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
        DUTY_STAFF_PREFERENCE: "Teachers",
    }
    scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": teacher_queue, "Temps": temp_queue}, required_count=1, ideal_case=False)

    assert len(duty_info["assignees"]) == 1
    assert duty_info["assignees"][0].get_name() == "Teacher"


def test_scheduler_assign_staff_to_duty_defaults_to_no_preference_when_missing():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
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
    scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": teacher_queue, "Temps": temp_queue}, required_count=1, ideal_case=False)

    assert len(duty_info["assignees"]) == 1


def test_order_duties_for_assignment_prioritizes_constrained_duties():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
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

    teacher_queue = Queue()
    temp_queue = Queue()

    ordered = scheduler._order_duties_for_assignment(duties, {"Teachers": teacher_queue, "Temps": temp_queue}, scheduler._staff_attributes)
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
                ["Teacher", "Monday", "AM", "Alice"],
                ["Teacher", "Tuesday", "PM", "Bob"],
            ],
            columns=["Staff Type", "Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Sheet1", index=False)

        pd.DataFrame(
            [
                ["Temp", "Wednesday", "AM", "Carol"],
                ["Temp", "Thursday", "PM", "Dave"],
            ],
            columns=["Staff Type", "Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Sheet2", index=False)

        pd.DataFrame(
            [
                ["CH", "Friday", "AM", "Eve"],
            ],
            columns=["Staff Type", "Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Sheet3", index=False)

    result = Scheduler._get_staff_availability(str(file_path))

    assert len(result) == 3
    assert "Teacher" in result
    assert "Temp" in result
    assert "CH" in result

    teachers = result["Teacher"]
    temps = result["Temp"]
    ch = result["CH"]

    assert len(teachers) == 2
    assert len(temps) == 2
    assert len(ch) == 1

    assert teachers[0] == ["Teacher", "Monday", "AM", "Alice", 1.0]
    assert teachers[1] == ["Teacher", "Tuesday", "PM", "Bob", 1.0]

    assert temps[0] == ["Temp", "Wednesday", "AM", "Carol", 1.0]
    assert temps[1] == ["Temp", "Thursday", "PM", "Dave", 1.0]

    assert ch[0] == ["CH", "Friday", "AM", "Eve", 1.0]


def test_scheduler_get_staff_availability_combines_staff_types_across_sheets(tmp_path):
    file_path = tmp_path / "availability.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Teacher", "Monday", "AM", "Alice"],
                ["Teacher", "Tuesday", "PM", "Bob"],
            ],
            columns=["Staff Type", "Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Sheet1", index=False)

        pd.DataFrame(
            [
                ["Teacher", "Wednesday", "AM", "Carol"],
                ["Temp", "Thursday", "PM", "Dave"],
            ],
            columns=["Staff Type", "Day", "Session", "Name"],
        ).to_excel(writer, sheet_name="Sheet2", index=False)

    result = Scheduler._get_staff_availability(str(file_path))

    assert len(result) == 2
    assert "Teacher" in result
    assert "Temp" in result

    assert result["Teacher"] == [
        ["Teacher", "Monday", "AM", "Alice", 1.0],
        ["Teacher", "Tuesday", "PM", "Bob", 1.0],
        ["Teacher", "Wednesday", "AM", "Carol", 1.0],
    ]

    assert result["Temp"] == [
        ["Temp", "Thursday", "PM", "Dave", 1.0],
    ]

    
def test_scheduler_get_duties_list_from_excel_populates_duty_roster(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Day": ["Monday"],
            "Date": ["2026-08-03"],
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
    day_key = "Monday_2026-08-03"

    assert day_key in duty_roster

    hall_duty = next((duty for duty in duty_roster[day_key].values() if duty.get("activity") == "Hall Duty"), None)
    assert hall_duty is not None

    assert "Hall Duty" in hall_duty.get("activity")
    assert hall_duty[DUTY_DURATION] == pytest.approx(1.0)
    assert hall_duty[DUTY_STAFF_PREFERENCE] == "Teacher First"
    assert hall_duty[DUTY_MIN_REQUIREMENT] == 1


def test_scheduler_get_duties_list_from_excel_normalizes_float_times(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Day": ["Monday"],
            "Date": ["2026-08-03"],
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
    day_key = "Monday_2026-08-03"

    assert day_key in duty_roster

    hall_duty = next((duty for duty in duty_roster[day_key].values() if duty.get("activity") == "Hall Duty"), None)
    assert hall_duty is not None

    assert "Hall Duty" in hall_duty.get("activity")
    assert hall_duty[DUTY_START_TIME] == "0800"
    assert hall_duty[DUTY_END_TIME] == "1700"
    assert hall_duty[DUTY_DURATION] == pytest.approx(9.0)
    assert hall_duty[DUTY_STAFF_PREFERENCE] == "Teacher First"


def test_scheduler_get_duties_list_from_excel_rejects_day_date_mismatch(tmp_path):
    file_path = tmp_path / "duties.xlsx"
    duties = pd.DataFrame(
        {
            "Day": ["Wednesday"],
            "Date": ["2026-09-15"],
            "Activity": ["Montessori Circle"],
            "Session": ["AM"],
            "Start Time": ["0900"],
            "End Time": ["0930"],
            "Minimum Requirement": [3],
            "Ideal Case": [3],
            "Required Function": ["Montessori"],
            "Restricted Function": [None],
            "Staff Preference": ["No Preference"],
        }
    )
    duties.to_excel(file_path, index=False)

    scheduler = object.__new__(Scheduler)
    scheduler._duty_roster = DutyRoster()

    with pytest.raises(ValueError, match="does not match.*expected Tuesday"):
        scheduler._get_duties_list_from_excel(str(file_path))


def test_scheduler_assign_staff_to_duty_fails_when_specialized_staff_is_misallocated(monkeypatch):
    # This test simulates a scenario where a multi-skilled person is assigned to a
    # general duty, making them unavailable for a specialized duty that only they can perform.
    # The scheduler should be smart enough to reserve them for the specialized duty.

    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"

    # Three teachers, Alice has a unique skill 'First Aid'
    scheduler._staff_attributes.add_required_function("Alice", "Structured Classes")
    scheduler._staff_attributes.add_required_function("Alice", "First Aid")
    scheduler._staff_attributes.add_required_function("Bob", "Structured Classes")
    scheduler._staff_attributes.add_required_function("Carol", "Structured Classes")

    teacher_queue = Queue()
    temp_queue = Queue()
    teacher_queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    teacher_queue.add_to_queue("Bob", "Monday", "0900", "1000", 0)
    teacher_queue.add_to_queue("Carol", "Monday", "0900", "1000", 0)

    # Two duties for the same time slot.
    # 'Structured Classes' needs 2 people, 'First Aid' needs 1.
    # Alice is the only one who can do 'First Aid'.
    duties = {
        "first_aid_duty": {
            DUTY_ACTIVITY: "First Aid Duty",
            DUTY_START_TIME: "0900",
            DUTY_END_TIME: "1000",
            DUTY_MIN_REQUIREMENT: 1,
            DUTY_IDEAL_CASE: 1,
            DUTY_REQUIRED_FUNCTION: "First Aid",
            DUTY_RESTRICTED_FUNCTION: None,
            DUTY_ASSIGNEES: [],
        },
        "structured_duty": {
            DUTY_ACTIVITY: "Structured Classes Duty",
            DUTY_START_TIME: "0900",
            DUTY_END_TIME: "1000",
            DUTY_MIN_REQUIREMENT: 2,
            DUTY_IDEAL_CASE: 2,
            DUTY_REQUIRED_FUNCTION: "Structured Classes",
            DUTY_RESTRICTED_FUNCTION: None,
            DUTY_ASSIGNEES: [],
        },
    }

    monkeypatch.setattr("planner.queue.shuffle", lambda x: None)

    # First attempt (bad order)
    _teacher_q_1 = copy.deepcopy(teacher_queue)
    _temp_q_1 = copy.deepcopy(temp_queue)
    ordered_duties = scheduler._order_duties_for_assignment(duties, {"Teachers": _teacher_q_1, "Temps": _temp_q_1}, scheduler._staff_attributes)

    # Manually assign the less-constrained 'structured_duty' first to simulate a bad assignment order.
    # With no shuffling, this will assign Alice and Bob.
    scheduler._assign_staff_to_duty("Monday", duties["structured_duty"], {"Teachers": _teacher_q_1, "Temps": _temp_q_1}, duties["structured_duty"][DUTY_MIN_REQUIREMENT], ideal_case=False, duty_name="Structured Classes Duty")

    # Now, attempt to assign the first_aid_duty. This should fail because Alice is no longer available.
    result_1 = scheduler._assign_staff_to_duty("Monday", duties["first_aid_duty"], {"Teachers": _teacher_q_1, "Temps": _temp_q_1}, 1, ideal_case=False, duty_name="First Aid Duty")
    assert isinstance(result_1, str)
    assert "Unable to find sufficient staff" in result_1

    # Second attempt (simulating a shuffle where Alice is last)
    _teacher_q_2 = copy.deepcopy(teacher_queue)
    _teacher_q_2._queue.reverse() # Move Alice to the end
    for duty_id, duty_info in ordered_duties:
        assert scheduler._assign_staff_to_duty("Monday", duty_info, {"Teachers": _teacher_q_2, "Temps": temp_queue}, duty_info[DUTY_MIN_REQUIREMENT], ideal_case=False, duty_name=duty_info[DUTY_ACTIVITY]) is True


def test_scheduler_optimize_duty_assignment_raises_value_error_on_failure():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._fairness_mode = "week"
    scheduler._lunch_break_start = "1200"
    scheduler._lunch_break_end = "1400"
    scheduler._lunch_break_min_rest_slots = 2
    scheduler._duty_roster = DutyRoster()

    # Define a duty roster with an impossible duty
    scheduler._duty_roster._add_day("Monday")
    scheduler._duty_roster.add_duty(
        day="Monday",
        activity="Supervision",
        session="AM",
        start_time="0900",
        end_time="1000",
        min_requirement=5,
        ideal_case=5,
        required_function=None,
        restricted_function=None,
        staff_preference="No Preference",
    )

    teacher_queue = Queue()
    temp_queue = Queue()
    # Add only 1 person when 5 are required
    teacher_queue.add_to_queue("Teacher1", "Monday", "0800", "1700", 0)

    with pytest.raises(ValueError) as exc_info:
        scheduler._optimize_duty_assignment({"Teachers": teacher_queue, "Temps": temp_queue})

    err_msg = str(exc_info.value)
    assert "Unable to find sufficient staff for Supervision" in err_msg
    assert "--- Target Duty Details ---" in err_msg
    assert "--- Concurrent Duties in Timeframe" in err_msg
    assert "--- Staff Status during Timeframe" in err_msg
    assert "Available & Engaged Staff" in err_msg


def test_scheduler_assign_with_three_staff_types():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._fairness_mode = "week"
    scheduler._lunch_break_start = "1200"
    scheduler._lunch_break_end = "1400"
    scheduler._lunch_break_min_rest_slots = 0
    scheduler._duty_roster = DutyRoster()

    scheduler._duty_roster._add_day("Monday")
    scheduler._duty_roster.add_duty(
        day="Monday",
        activity="Morning Duty",
        session="AM",
        start_time="0900",
        end_time="1000",
        min_requirement=3,
        ideal_case=3,
        required_function=None,
        restricted_function=None,
        staff_preference="No Preference",
    )

    teacher_q = Queue()
    temp_q = Queue()
    ch_q = Queue()
    teacher_q.add_to_queue("Alice", "Monday", "0800", "1700", 0)
    temp_q.add_to_queue("Bob", "Monday", "0800", "1700", 0)
    ch_q.add_to_queue("Carol", "Monday", "0800", "1700", 0)

    staff_queues = {
        "Teachers": teacher_q,
        "Temps": temp_q,
        "CH": ch_q,
    }

    state = scheduler._optimize_duty_assignment(staff_queues)
    assert state is not None
    assert len(state.staff_queues) == 3
    all_people = state.get_all_people()
    assert len(all_people) == 3
    duty_info = list(state.roster["Monday"].values())[0]
    assignees = duty_info["assignees"]
    assert len(assignees) == 3
    assignee_names = {a.get_name() for a in assignees}
    assert assignee_names == {"Alice", "Bob", "Carol"}


def test_scheduler_preference_order_chained_types():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"

    t_q = Queue()
    temp_q = Queue()
    ch_q = Queue()
    t_q.add_to_queue("Teacher1", "Monday", "0900", "1000", 0)
    temp_q.add_to_queue("Temp1", "Monday", "0900", "1000", 0)
    ch_q.add_to_queue("CH1", "Monday", "0900", "1000", 0)

    staff_queues = {
        "Teachers": t_q,
        "Temps": temp_q,
        "CH": ch_q,
    }

    # Test preference: 'CH; temps; teacher'
    ordered = scheduler._order_queues_by_preference(staff_queues, "CH; temps; teacher")
    assert ordered == [ch_q, temp_q, t_q]

    # Test preference: 'temps:CH:teacher'
    ordered2 = scheduler._order_queues_by_preference(staff_queues, "temps:CH:teacher")
    assert ordered2 == [temp_q, ch_q, t_q]

    # Test preference: 'teacher; CH' (temps omitted, should follow at end)
    ordered3 = scheduler._order_queues_by_preference(staff_queues, "teacher; CH")
    assert ordered3 == [t_q, ch_q, temp_q]


def test_scheduler_preference_invalid_type_raises_value_error():
    scheduler = object.__new__(Scheduler)
    staff_queues = {
        "Teachers": Queue(),
        "Temps": Queue(),
        "CH": Queue(),
    }

    with pytest.raises(ValueError) as exc_info:
        scheduler._order_queues_by_preference(staff_queues, "Astronaut; Temps")

    err = str(exc_info.value)
    assert "Invalid staff type 'Astronaut'" in err
    assert "Valid staff types are: ['Teachers', 'Temps', 'CH', 'No Preference']" in err


def test_scheduler_add_to_queue_reads_expected_capacity():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    staff_list = [
        ["Monday", "AM", "Alice", 900, 1200, "Teacher", 0.5],
        ["Monday", "AM", "Bob", 900, 1200, "Teacher", 1.0],
    ]

    scheduler._add_to_queue(queue, staff_list)

    people = {person.get_name(): person for person in queue.get_list()}
    assert people["Alice"].get_expected_capacity("Monday_AM") == 0.5
    assert people["Bob"].get_expected_capacity("Monday_AM") == 1.0


def test_scheduler_add_to_queue_defaults_missing_expected_capacity_to_one():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    staff_list = [
        ["Monday", "AM", "Alice", 900, 1200],
    ]

    scheduler._add_to_queue(queue, staff_list)

    assert queue.get_list()[0].get_expected_capacity() == 1.0


def test_scheduler_add_to_queue_keeps_expected_capacity_per_day():
    scheduler = object.__new__(Scheduler)
    queue = Queue()

    staff_list = [
        ["Monday", "AM", "Ferninda", 900, 1200, "Teacher", 0.7],
        ["Tuesday", "AM", "Ferninda", 900, 1200, "Teacher", 0.5],
    ]

    scheduler._add_to_queue(queue, staff_list)

    person = queue.get_list()[0]
    assert person.get_expected_capacity("Monday_AM") == 0.7
    assert person.get_expected_capacity("Tuesday_AM") == 0.5
    assert person.get_expected_capacity() == pytest.approx(0.6)


def test_scheduler_parse_expected_capacity_defaults_blank_values():
    assert Scheduler._parse_expected_capacity(None) == 1.0
    assert Scheduler._parse_expected_capacity("") == 1.0
    assert Scheduler._parse_expected_capacity(float("nan")) == 1.0
    assert Scheduler._parse_expected_capacity("0.5") == 0.5


def test_scheduler_get_staff_availability_reads_expected_capacity(tmp_path):
    file_path = tmp_path / "availability.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Monday", "AM", "Alice", 900, 1200, "Teacher", 0.5],
                ["Monday", "AM", "Bob", 900, 1200, "Teacher", 1.0],
            ],
            columns=[
                "Day",
                "Session",
                "Name",
                "Start Time",
                "End Time",
                "Staff Type",
                "Expected Capacity",
            ],
        ).to_excel(writer, sheet_name="Staff", index=False)

    result = Scheduler._get_staff_availability(str(file_path))

    assert result["Teacher"][0][-1] == 0.5
    assert result["Teacher"][1][-1] == 1.0


def test_scheduler_assigns_more_work_to_higher_expected_capacity():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
    scheduler._lunch_break_min_rest_slots = 0

    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1200", 0, expected_capacity=0.5)
    queue.add_to_queue("Bob", "Monday", "0900", "1200", 0, expected_capacity=1.0)

    for start, end in (("0900", "0930"), ("0930", "1000"), ("1000", "1030")):
        duty_info = {
            DUTY_START_TIME: start,
            DUTY_END_TIME: end,
            DUTY_ASSIGNEES: [],
            DUTY_REQUIRED_FUNCTION: None,
            DUTY_RESTRICTED_FUNCTION: None,
            DUTY_STAFF_PREFERENCE: None,
        }
        result = scheduler._assign_staff_to_duty(
            "Monday",
            duty_info,
            {"Teachers": queue},
            required_count=1,
            ideal_case=False,
        )
        assert result is True

    people = {person.get_name(): person for person in queue.get_list()}
    assert people["Alice"].get_hours_worked() == 0.5
    assert people["Bob"].get_hours_worked() == 1.0
    assert people["Bob"].get_hours_worked() == 2 * people["Alice"].get_hours_worked()


def test_scheduler_optimize_distributes_work_by_expected_capacity():
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._fairness_mode = "week"
    scheduler._lunch_break_start = "0000"
    scheduler._lunch_break_end = "0000"
    scheduler._lunch_break_min_rest_slots = 0
    scheduler._duty_roster = DutyRoster()
    scheduler._duty_roster._add_day("Monday")

    for start, end in (
        ("0900", "0930"),
        ("0930", "1000"),
        ("1000", "1030"),
        ("1030", "1100"),
        ("1100", "1130"),
        ("1130", "1200"),
    ):
        scheduler._duty_roster.add_duty(
            day="Monday",
            activity=f"Duty {start}",
            session="AM",
            start_time=start,
            end_time=end,
            min_requirement=1,
            ideal_case=1,
            required_function=None,
            restricted_function=None,
            staff_preference="No Preference",
        )

    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1200", 0, expected_capacity=0.5)
    queue.add_to_queue("Bob", "Monday", "0900", "1200", 0, expected_capacity=1.0)

    state = scheduler._optimize_duty_assignment({"Teachers": queue})
    people = {person.get_name(): person for person in state.get_all_people()}

    assert people["Alice"].get_hours_worked() == 1.0
    assert people["Bob"].get_hours_worked() == 2.0
    assert people["Bob"].get_hours_worked() == 2 * people["Alice"].get_hours_worked()
    assert people["Alice"].get_work_capacity_ratio() == pytest.approx(
        people["Bob"].get_work_capacity_ratio()
    )




