from typing import Any, Dict, List


class ScheduleState:
    """Represents the schedule state including duty roster and staff queues."""

    def __init__(
        self,
        roster: dict,
        staff_queues: Dict[str, Any],
    ):
        self.roster = roster
        self.staff_queues = dict(staff_queues)

    def get_all_people(self) -> List[Any]:
        """Return a flat list of all staff members from all queues."""
        people = []
        for q in self.staff_queues.values():
            if hasattr(q, "get_list"):
                people.extend(q.get_list())
            elif isinstance(q, list):
                people.extend(q)
        return people


