import unittest
import tempfile
import os

from planner.schedule_state import ScheduleState
from planner.state_serializer import StateSerializer
from planner.queue import Queue


class TestScheduleStateAndSerializer(unittest.TestCase):

    def test_schedule_state_constructs_and_preserves_queues(self):
        roster = {"Day_AM": {"Duty": {"start_time": "0900", "end_time": "0930", "assignees": []}}}
        teachers = Queue()
        temps = Queue()
        teachers.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)
        temps.add_to_queue("Temp1", "Day_AM", "0900", "1700", 0)

        state = ScheduleState(roster, teachers, temps)
        self.assertIs(state.roster, roster)
        self.assertIs(state.teachers, teachers)
        self.assertIs(state.temps, temps)

    def test_state_serializer_roundtrip(self):
        roster = {"Day_AM": {"Duty": {"start_time": "0900", "end_time": "0930", "assignees": []}}}
        teachers = Queue()
        temps = Queue()
        teachers.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)
        temps.add_to_queue("Temp1", "Day_AM", "0900", "1700", 0)

        state = ScheduleState(roster, teachers, temps)

        fd, path = tempfile.mkstemp(suffix=".state")
        os.close(fd)
        try:
            StateSerializer.save(state, path)
            loaded = StateSerializer.load(path)

            # Basic equality checks
            self.assertEqual(loaded.roster, state.roster)
            self.assertEqual([p.get_name() for p in loaded.teachers.get_list()], [p.get_name() for p in state.teachers.get_list()])
            self.assertEqual([p.get_name() for p in loaded.temps.get_list()], [p.get_name() for p in state.temps.get_list()])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
