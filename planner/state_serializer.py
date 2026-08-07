import pickle
from typing import Any

from .schedule_state import ScheduleState


class StateSerializer:
    """Serialize and deserialize ScheduleState objects.

    Uses pickle by default; API isolates persistence so implementation can
    be changed to JSON later with minimal changes.
    """

    @staticmethod
    def save(schedule_state: ScheduleState, filename: str) -> None:
        with open(filename, "wb") as fh:
            pickle.dump(schedule_state, fh)

    @staticmethod
    def load(filename: str) -> ScheduleState:
        with open(filename, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, ScheduleState):
            raise TypeError("Loaded object is not a ScheduleState")
        return obj
