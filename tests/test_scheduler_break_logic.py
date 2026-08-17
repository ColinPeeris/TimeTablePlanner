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
        "Monday", duty_info, teacher_queue, Queue(), required_count=1, ideal_case=False
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
        "Monday", duty_info, teacher_queue, Queue(), required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Bob"


def test_ignores_rested_preference_outside_lunch_window(scheduler_with_lunch_break):
    """
    Verify that outside the lunch window, the rested-staff preference is ignored,
    and the first person in the queue is chosen.
    """
    # Bob was working 10:30-11:00, Alice was resting.
    # The duty starts at 11:00, outside the lunch window.
    # Bob is first in the queue, so he should be chosen.
    teacher_queue = Queue()
    teacher_queue.add_to_queue("Bob", "Monday", "1030", "1100", 1) # Working before
    teacher_queue.add_to_queue("Bob", "Monday", "1100", "1200", 0) # Available for duty
    teacher_queue.add_to_queue("Alice", "Monday", "1030", "1200", 0) # Resting before and available

    duty_info = {
        DUTY_START_TIME: "1100",
        DUTY_END_TIME: "1130",
        DUTY_ASSIGNEES: [],
    }

    result = scheduler_with_lunch_break._assign_staff_to_duty(
        "Monday", duty_info, teacher_queue, Queue(), required_count=1, ideal_case=False
    )

    assert result is True
    assert len(duty_info[DUTY_ASSIGNEES]) == 1
    assert duty_info[DUTY_ASSIGNEES][0].get_name() == "Bob"