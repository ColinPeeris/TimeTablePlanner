import os
import tempfile
import unittest

from planner.schedule_state import ScheduleState
from planner.state_serializer import StateSerializer
from planner.queue import Queue
from report_generator import ReportGenerator


class TestReportGeneratorStateIntegration(unittest.TestCase):
    def test_report_generator_accepts_schedule_state_and_writes_excel(self):
        roster = {"Day_AM": {"Duty": {"start_time": "0900", "end_time": "0930", "assignees": []}}}
        teachers = Queue()
        temps = Queue()
        teachers.add_to_queue("Alice", "Day_AM", "0900", "1700", 0)
        temps.add_to_queue("Temp1", "Day_AM", "0900", "1700", 0)

        state = ScheduleState(roster, {"Teachers": teachers, "Temps": temps})

        fd, xlsx_path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        try:
            ReportGenerator(state, filename=xlsx_path).generate()
            self.assertTrue(os.path.exists(xlsx_path))
            self.assertGreater(os.path.getsize(xlsx_path), 0)
        finally:
            try:
                os.remove(xlsx_path)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()