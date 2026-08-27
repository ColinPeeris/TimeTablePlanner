import os
import tempfile
import unittest

from planner.scheduler import Scheduler
from planner.schedule_state import ScheduleState
from planner.queue import Queue
from planner.staff_attributes import StaffAttributes
from planner.duty_roster import DutyRoster
from planner.state_serializer import StateSerializer


class DummyDutyRoster:
    def __init__(self, roster):
        self._roster = roster

    def get_duty_roster(self):
        return self._roster


class TestSchedulerStateGeneration(unittest.TestCase):

    def _make_scheduler_obj(self):
        scheduler = object.__new__(Scheduler)
        scheduler._staff_attributes = StaffAttributes()
        # use small lunch requirement so check passes
        scheduler._lunch_break_start = "0000"
        scheduler._lunch_break_end = "2359"
        scheduler._lunch_break_min_rest_slots = 0
        scheduler._fairness_mode = "week"
        return scheduler

    def test_optimize_returns_schedule_state(self):
        scheduler = self._make_scheduler_obj()

        # Build simple duty roster: one day, one duty
        duty_roster = {
            "Day_AM": {
                "Hall Duty": {
                    "start_time": "0900",
                    "end_time": "1000",
                    "min_requirement": 1,
                    "ideal_case": 1,
                    "assignees": [],
                }
            }
        }

        scheduler._duty_roster = DummyDutyRoster(duty_roster)

        # Create queues
        teacher_q = Queue()
        temp_q = Queue()
        teacher_q.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)

        staff_queues = {"Teachers": teacher_q, "Temps": temp_q}
        state = scheduler._optimize_duty_assignment(staff_queues)
        self.assertIsNotNone(state)
        self.assertIsInstance(state, ScheduleState)
        self.assertIn("Day_AM", state.roster)
        self.assertEqual(len(state.staff_queues["Teachers"].get_list()), 1)

    def test_save_and_load_state_file(self):
        scheduler = self._make_scheduler_obj()

        duty_roster = {
            "Day_AM": {
                "Hall Duty": {
                    "start_time": "0900",
                    "end_time": "1000",
                    "min_requirement": 1,
                    "ideal_case": 1,
                    "assignees": [],
                }
            }
        }
        scheduler._duty_roster = DummyDutyRoster(duty_roster)

        teacher_q = Queue()
        temp_q = Queue()
        teacher_q.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)

        staff_queues = {"Teachers": teacher_q, "Temps": temp_q}
        state = scheduler._optimize_duty_assignment(staff_queues)

        fd, path = tempfile.mkstemp(suffix=".state")
        os.close(fd)
        try:
            StateSerializer.save(state, path)
            loaded = StateSerializer.load(path)

            # Compare roster structure (primitive fields) and assignee names
            self.assertEqual(sorted(loaded.roster.keys()), sorted(state.roster.keys()))
            for day in state.roster:
                self.assertIn(day, loaded.roster)
                for duty_name, duty_info in state.roster[day].items():
                    self.assertIn(duty_name, loaded.roster[day])
                    loaded_info = loaded.roster[day][duty_name]
                    # Compare primitive fields
                    for key in ("start_time", "end_time", "min_requirement", "ideal_case"):
                        self.assertEqual(loaded_info.get(key), duty_info.get(key))
                    # Compare assignee names
                    loaded_assignees = [a.get_name() for a in loaded_info.get("assignees", [])]
                    orig_assignees = [a.get_name() for a in duty_info.get("assignees", [])]
                    self.assertEqual(loaded_assignees, orig_assignees)

            self.assertEqual([p.get_name() for p in loaded.staff_queues["Teachers"].get_list()], [p.get_name() for p in state.staff_queues["Teachers"].get_list()])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()