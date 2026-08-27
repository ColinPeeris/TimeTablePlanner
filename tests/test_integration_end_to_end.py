import os
import tempfile
import unittest

from planner.schedule_state import ScheduleState
from planner.state_serializer import StateSerializer
from planner.queue import Queue
from report_generator import ReportGenerator


class TestEndToEndIntegration(unittest.TestCase):

    def test_end_to_end_generate_and_reload(self):
        roster = {"Day_AM": {"Hall Duty": {"start_time": "0900", "end_time": "1000", "assignees": []}}}
        teachers = Queue()
        temps = Queue()
        teachers.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)
        temps.add_to_queue("Temp1", "Day_AM", "0900", "1700", 0)

        state = ScheduleState(roster, {"Teachers": teachers, "Temps": temps})

        fd1, xlsx_path = tempfile.mkstemp(suffix=".xlsx")
        fd2, state_path = tempfile.mkstemp(suffix=".state")
        os.close(fd1)
        os.close(fd2)
        try:
            # generate workbook
            ReportGenerator(state, filename=xlsx_path).generate()
            self.assertTrue(os.path.exists(xlsx_path))

            # save and load state
            StateSerializer.save(state, state_path)
            loaded = StateSerializer.load(state_path)
            self.assertEqual(loaded.roster, state.roster)

            # regenerate workbook from loaded state
            fd3, xlsx2 = tempfile.mkstemp(suffix=".xlsx")
            os.close(fd3)
            ReportGenerator(loaded, filename=xlsx2).generate()
            self.assertTrue(os.path.exists(xlsx2))
        finally:
            for p in (xlsx_path, state_path, xlsx2):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == '__main__':
    unittest.main()
