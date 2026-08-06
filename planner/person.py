from typing import List

import numpy as np
from dataclasses import dataclass
from .utils.constants import DUTY_END_TIME, DUTY_START_TIME

@dataclass(frozen=True)
class ScheduleConfig:
    """Configuration for a person's daily schedule grid.

    Attributes:
        start_time (str): Earliest schedule time in HHMM format.
        end_time (str): Latest schedule time in HHMM format.
        slot_minutes (int): Length of each schedule slot in minutes.
    """

    start_time: str = "0900"
    end_time: str = "1800"
    slot_minutes: int = 30

    @property
    def start_minutes(self):
        """Return the start time converted to minutes from midnight."""
        h, m = divmod(int(self.start_time), 100)
        return h * 60 + m

    @property
    def end_minutes(self):
        """Return the end time converted to minutes from midnight."""
        h, m = divmod(int(self.end_time), 100)
        return h * 60 + m

    @property
    def num_slots(self):
        """Return the number of schedule slots within the configured day."""
        return (self.end_minutes - self.start_minutes) // self.slot_minutes


class Person:
    """
    A class representing a person (e.g., a teacher or temp) who can be assigned duties based on availability.

    Attributes:
        _name (str): The name of the person.
        _availability_by_hour (dict): The person's availability by day and 30-minute slot.
        _days_assigned (list): Days when the person has duties assigned.
    """

    def __init__(self, name: str, config: ScheduleConfig = ScheduleConfig()):
        """Create a new Person with an availability schedule.

        Args:
            name (str): The person's name.
            config (ScheduleConfig): Schedule bounds and slot duration.
        """
        self._name = name
        self._config = config
        self._availability_by_hour = {}
        self._days_assigned = []
        self._duties_by_day = {}

        self._base_start_minutes = self._config.start_minutes
        self._slot_minutes = self._config.slot_minutes

    @staticmethod
    def normalize_time(time_value):
        """Normalize time values to integer HHMM format.

        Accepts integers, floats representing whole HHMM values, and strings.
        Returns an integer suitable for schedule calculations.
        """
        if isinstance(time_value, float):
            if not time_value.is_integer():
                raise ValueError(
                    f"Time values must be whole minutes in HHMM format, not {time_value}"
                )
            time_value = int(time_value)

        if isinstance(time_value, str):
            time_value = time_value.strip()
            if "." in time_value:
                numeric = float(time_value)
                if not numeric.is_integer():
                    raise ValueError(
                        f"Time values must be whole minutes in HHMM format, not {time_value}"
                    )
                time_value = int(numeric)
            return int(time_value)

        if isinstance(time_value, int):
            return time_value

        raise TypeError(
            f"Unsupported time type: {type(time_value).__name__}"
        )

    @staticmethod
    def time_to_index(time, base_start_minutes: int = None, slot_minutes: int = 30):
        """Convert an HHMM time value into a schedule slot index.

        Args:
            time: A time value in HHMM format (int, float, or str).
            base_start_minutes (int): Base minutes used for index zero.
            slot_minutes (int): Slot duration in minutes.

        Returns:
            int: The zero-based slot index for the requested time.
        """
        time_value = Person.normalize_time(time)
        h, m = divmod(int(time_value), 100)
        minutes = h * 60 + m
        if base_start_minutes is None:
            base_start_minutes = int("0900"[:2]) * 60 + int("0900"[2:])
        return (minutes - base_start_minutes) // slot_minutes

    def _prepend_slots(self, slot_count: int) -> None:
        """Prepend placeholder slots when availability begins before the configured start."""
        if slot_count <= 0:
            return
        for day in self._availability_by_hour:
            self._availability_by_hour[day] = [-1] * slot_count + self._availability_by_hour[day]
        self._base_start_minutes -= slot_count * self._slot_minutes

    def get_name(self) -> str:
        """Return the person's name."""
        return self._name

    def get_availability(self, day: str) -> List[int]:
        """Return the availability array for a specific day."""
        return self._availability_by_hour.get(day, [])

    def get_work_capacity_ratio(self) -> float:
        """Return the ratio of filled work slots to total school-day slots."""
        total_filled_slots = 0
        total_free_slots = 0
        for day in self._availability_by_hour:
            total_filled_slots += self._availability_by_hour[day].count(1)
            total_free_slots += self._availability_by_hour[day].count(0)
        return float(total_filled_slots) / (total_filled_slots + total_free_slots) \
            if (total_filled_slots + total_free_slots) > 0 else 0.0

    def get_hours_worked(self):
        return sum(self.get_hours_worked_by_day().values())

    def get_hours_worked_by_day(self):
        """Return a dictionary of hours worked per day."""
        hours = {}
        for day, slots in self._availability_by_hour.items():
            hours[day] = slots.count(1) * self._slot_minutes / 60
        return hours

    def get_rest_periods_by_day(self):
        """Return a dictionary of rest periods per day."""
        rest_periods = {}
        for day, slots in self._availability_by_hour.items():
            rest_periods[day] = slots.count(0) * self._slot_minutes / 60
        return rest_periods

    def get_hours_in_school(self) -> float:
        """Return the total hours the person is present in school, including free slots."""
        total_slots_in_school = 0
        for day in self._availability_by_hour:
            total_slots_in_school += self._availability_by_hour[day].count(0) + self._availability_by_hour[day].count(1)
        return float(total_slots_in_school) / 2

    def set_availability(self, day: str, start_time: str, end_time: str, status: int) -> None:
        """Mark a person's availability status for a time range on a given day.

        Args:
            day (str): The day key used in the schedule, such as 'Monday_AM'.
            start_time (str): Start time in HHMM format.
            end_time (str): End time in HHMM format.
            status (int): Availability status (-1 for unavailable, 0 for free, 1 for filled).
        """
        if day not in self._availability_by_hour:
            self._availability_by_hour[day] = [-1] * self._config.num_slots
        start_index = self.time_to_index(start_time, self._base_start_minutes, self._slot_minutes)
        end_index = self.time_to_index(end_time, self._base_start_minutes, self._slot_minutes)
        if start_index < 0:
            self._prepend_slots(-start_index)
            start_index = self.time_to_index(start_time, self._base_start_minutes, self._slot_minutes)
            end_index = self.time_to_index(end_time, self._base_start_minutes, self._slot_minutes)
        if end_index > len(self._availability_by_hour[day]):
            self._availability_by_hour[day].extend([-1] * (end_index - len(self._availability_by_hour[day])))
        for i in range(start_index, end_index):
            self._availability_by_hour[day][i] = status

    def check_availability(self, day: str, start_time: str, end_time: str) -> bool:
        """Check whether the person is free for the given time range on a day."""
        start_index = self.time_to_index(start_time, self._base_start_minutes, self._slot_minutes)
        end_index = self.time_to_index(end_time, self._base_start_minutes, self._slot_minutes)
        if day not in self._availability_by_hour:
            return False
        if start_index < 0 or end_index > len(self._availability_by_hour[day]):
            return False
        return (np.asarray(self._availability_by_hour[day][start_index:end_index]) == 0).all()

    def add_duty(self, day, duty_name=None, duty_info: dict = None):

        # Backwards-compatible behaviour: if only `day` is provided,
        # treat this as adding the day to the assigned days list.
        if duty_name is None:
            if day not in self._days_assigned:
                self._days_assigned.append(day)
            return

        if duty_info is None:
            duty_info = {}

        if day not in self._duties_by_day:
            self._duties_by_day[day] = []

        self._duties_by_day[day].append(
            {
                "name": duty_name,
                **duty_info,
            }
        )

    def get_duties(self, day: str):
        """
        Return all duties assigned on a particular day.
        """
        return self._duties_by_day.get(day, [])


    def get_activity_at(self, day: str, time: str) -> str:
        """
        Return what the person is doing at a given time.

        Returns one of:
            Duty name
            "Rest"
            ""
        """

        slot = self.time_to_index(
            time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        if day not in self._availability_by_hour:
            return ""

        availability = self._availability_by_hour[day]

        if slot >= len(availability):
            return ""

        status = availability[slot]

        if status == -1:
            return ""

        if status == 0:
            return "Rest"

        for duty in self.get_duties(day):

            start = self.time_to_index(
                duty[DUTY_START_TIME],
                self._base_start_minutes,
                self._slot_minutes,
            )

            end = self.time_to_index(
                duty[DUTY_END_TIME],
                self._base_start_minutes,
                self._slot_minutes,
            )

            if start <= slot < end:
                return duty["name"]

        return "Duty"

    def build_daily_schedule(self, day: str):
        """
        Build a half-hour schedule for the specified day.

        Returns:
            list[dict]
        """

        schedule = []

        availability = self.get_availability(day)

        for slot, status in enumerate(availability):

            minutes = self._base_start_minutes + slot * self._slot_minutes

            hh = minutes // 60
            mm = minutes % 60

            start_time = f"{hh:02}{mm:02}"

            minutes += self._slot_minutes

            hh = minutes // 60
            mm = minutes % 60

            end_time = f"{hh:02}{mm:02}"

            activity = ""

            if status == -1:
                activity = ""

            elif status == 0:
                activity = "Rest"

            else:
                activity = "Duty"

                for duty in self.get_duties(day):

                    if (
                        duty[DUTY_START_TIME] <= start_time
                        and duty[DUTY_END_TIME] > start_time
                    ):
                        activity = duty["name"]
                        break

            schedule.append({
                "start": start_time,
                "end": end_time,
                "activity": activity,
            })

        return schedule