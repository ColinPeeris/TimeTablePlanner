from dataclasses import dataclass
from typing import Any


@dataclass
class ScheduleState:
    roster: dict
    teachers: Any
    temps: Any
