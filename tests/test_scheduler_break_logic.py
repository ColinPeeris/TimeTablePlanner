import pytest

from planner.scheduler import Scheduler
from planner.staff_attributes import StaffAttributes
from planner.queue import Queue
from planner.utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_START_TIME,
    DUTY_END_TIME,
)


@pytest.fixture
def scheduler_with_lunch_break():
    """Provides a Scheduler instance with a configured lunch break window."""
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "1200"
    scheduler._lunch_break_end = "1400"
    return scheduler


def test_prefers_rested_staff_during_lunch_window(scheduler_with_lunch_break):
    """
    Verify that during the lunch window, a staff member who was resting is
    chosen over one who was working, even if the working one is first in the queue.
    """
    # Bob was working 11:30-12:00, Alice was resting.
    # The duty starts at 12:00, during the lunch window.
    # Alice should be preferred.
    teacher_queue = Queue()
    teacher_queue.add_to_queue("Bob", "Monday", "1130", "1200", 1) # Working before
    teacher_queue.add_to_queue("Bob", "Monday", "1200", "1300", 0) # Available for duty
    teacher_queue.add_to_queue("Alice", "Monday", "1130", "1300", 0) # Resting before and available

    duty_info = {
        DUTY_START_TIME: "1200",
        DUTY_END_TIME: "1230",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler_with_lunch_break._assign_staff_to_duty(
        "Monday", duty_info, {"Teachers": teacher_queue}, required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Alice"


def test_fallback_to_non_rested_staff_during_lunch_window(scheduler_with_lunch_break):
    """
    Verify that if no rested staff are available during the lunch window,
    the scheduler falls back to assigning a non-rested (but available) staff member.
    """
    # Bob was working 11:30-12:00. Alice is unavailable for the duty.
    # The scheduler should fall back and assign Bob.
    teacher_queue = Queue()
    teacher_queue.add_to_queue("Bob", "Monday", "1130", "1200", 1) # Working before
    teacher_queue.add_to_queue("Bob", "Monday", "1200", "1300", 0) # Available for duty
    teacher_queue.add_to_queue("Alice", "Monday", "1130", "1200", 0) # Unavailable for duty

    duty_info = {
        DUTY_START_TIME: "1200",
        DUTY_END_TIME: "1230",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler_with_lunch_break._assign_staff_to_duty(
        "Monday", duty_info, {"Teachers": teacher_queue}, required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Bob"


def test_ignores_rested_preference_outside_lunch_window(scheduler_with_lunch_break):
    """
    Verify that outside the lunch window, the rested-staff preference is ignored.

    Alice already has more work, so capacity-aware selection prefers Bob even
    though he was working immediately before the slot.
    """
    teacher_queue = Queue()
    teacher_queue.add_to_queue("Bob", "Monday", "1030", "1100", 1)  # Working before
    teacher_queue.add_to_queue("Bob", "Monday", "1100", "1200", 0)  # Available for duty
    teacher_queue.add_to_queue("Alice", "Monday", "0900", "1100", 1)  # Already more work
    teacher_queue.add_to_queue("Alice", "Monday", "1100", "1200", 0)  # Available for duty

    duty_info = {
        DUTY_START_TIME: "1100",
        DUTY_END_TIME: "1130",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler_with_lunch_break._assign_staff_to_duty(
        "Monday", duty_info, {"Teachers": teacher_queue}, required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Bob"


def test_prefers_staff_who_will_not_exhaust_lunch_rest_budget():
    """
    Verify that a staff member who has plenty of lunch rest slots is preferred over
    a staff member who would exhaust their remaining lunch rest budget.
    """
    scheduler = object.__new__(Scheduler)
    scheduler._staff_attributes = StaffAttributes()
    scheduler._lunch_break_start = "1100"
    scheduler._lunch_break_end = "1300" # 4 slots: 1100, 1130, 1200, 1230
    scheduler._lunch_break_min_rest_slots = 1

    teacher_queue = Queue()
    # Emily is already working 1100-1230 (3 slots), so only 1 rest slot (1230-1300) remains.
    # Assigning 1230-1300 to Emily would reduce her rest slots to 0 (< 1).
    teacher_queue.add_to_queue("Emily", "Monday", "1100", "1230", 1)
    teacher_queue.add_to_queue("Emily", "Monday", "1230", "1300", 0)

    # Ferninda is resting 1100-1300 (4 rest slots).
    teacher_queue.add_to_queue("Ferninda", "Monday", "1100", "1300", 0)

    duty_info = {
        DUTY_START_TIME: "1230",
        DUTY_END_TIME: "1300",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler._assign_staff_to_duty(
        "Monday", duty_info, {"Teachers": teacher_queue}, required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    # Ferninda should be chosen instead of Emily to protect Emily's lunch rest slot
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Ferninda"


def test_lunch_rest_preference_overrides_expected_capacity(scheduler_with_lunch_break):
    """
    Verify that lunch-rest protection still wins when the working staff member
    has a higher expected capacity.
    """
    teacher_queue = Queue()
    teacher_queue.add_to_queue(
        "Bob", "Monday", "1130", "1200", 1, expected_capacity=1.0
    )
    teacher_queue.add_to_queue(
        "Bob", "Monday", "1200", "1300", 0, expected_capacity=1.0
    )
    teacher_queue.add_to_queue(
        "Alice", "Monday", "1130", "1300", 0, expected_capacity=0.5
    )

    duty_info = {
        DUTY_START_TIME: "1200",
        DUTY_END_TIME: "1230",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler_with_lunch_break._assign_staff_to_duty(
        "Monday", duty_info, {"Teachers": teacher_queue}, required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Alice"
