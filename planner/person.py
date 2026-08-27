from typing import List

import numpy as np

from .utils.configs import (
    SCHEDULE_START,
    SCHEDULE_END,
    SCHEDULE_SLOT_MINUTES,
)
from .utils.constants import DUTY_END_TIME, DUTY_START_TIME


class Person:
    """
    A class representing a person (e.g., a teacher or temp) who can be
    assigned duties based on availability.

    The person's availability is stored in configurable time slots.
    Schedule settings such as the start time, end time, and slot duration
    are loaded from the central configuration.
    """

    def __init__(self, name: str, staff_type: str = None):
        self._name = name
        self._staff_type = staff_type
        self._availability_by_hour = {}
        self._days_assigned = []
        self._duties_by_day = {}

        # Use the central schedule configuration.
        self._base_start_minutes = self._time_to_minutes(SCHEDULE_START)
        self._end_minutes = self._time_to_minutes(SCHEDULE_END)
        self._slot_minutes = SCHEDULE_SLOT_MINUTES

    @staticmethod
    def normalize_time(time_value) -> int:
        """Normalize a time-like value to an integer HHMM value."""
        if time_value is None:
            raise TypeError("Time value cannot be None")

        if isinstance(time_value, str):
            value = time_value.strip()
            if not value:
                raise ValueError("Time value cannot be empty")
            if ":" in value:
                hours, minutes = value.split(":", 1)
                value = f"{int(hours):02d}{int(minutes):02d}"
            if "." in value:
                numeric = float(value)
                if not numeric.is_integer():
                    raise ValueError(
                        f"Time values must be whole minutes in HHMM format, not {time_value}"
                    )
                value = str(int(numeric))
            return int(value)

        if isinstance(time_value, float):
            if not time_value.is_integer():
                raise ValueError(
                    f"Time values must be whole minutes in HHMM format, not {time_value}"
                )
            return int(time_value)

        if isinstance(time_value, int):
            return time_value

        raise TypeError(f"Unsupported time type: {type(time_value).__name__}")

    @staticmethod
    def _time_to_minutes(time_value) -> int:
        """
        Convert an HHMM time value into minutes from midnight.

        Accepts integers, floats representing whole HHMM values, and strings.
        """
        time_value = Person.normalize_time(time_value)
        hours, minutes = divmod(time_value, 100)
        return hours * 60 + minutes

    @classmethod
    def time_to_index(cls, time, base_start_minutes=None, slot_minutes=None):
        """
        Convert an HHMM time value into a schedule slot index.

        Args:
            time: Time in HHMM format.
            base_start_minutes: Optional schedule start in minutes.
            slot_minutes: Optional slot duration.

        Returns:
            int: Zero-based schedule slot index.
        """
        if base_start_minutes is None:
            base_start_minutes = cls._time_to_minutes(SCHEDULE_START)

        if slot_minutes is None:
            slot_minutes = SCHEDULE_SLOT_MINUTES

        minutes = cls._time_to_minutes(time)

        return (minutes - base_start_minutes) // slot_minutes

    def _get_number_of_slots(self) -> int:
        """Return the number of configured schedule slots."""
        duration = self._end_minutes - self._base_start_minutes

        if duration < 0:
            raise ValueError(
                "Schedule end time must be later than schedule start time."
            )

        if duration % self._slot_minutes != 0:
            raise ValueError(
                "Schedule duration must be evenly divisible by "
                "SCHEDULE_SLOT_MINUTES."
            )

        return duration // self._slot_minutes

    def _prepend_slots(self, slot_count: int) -> None:
        """
        Prepend unavailable slots when availability begins before the
        configured schedule start.
        """
        if slot_count <= 0:
            return

        for day in self._availability_by_hour:
            self._availability_by_hour[day] = (
                [-1] * slot_count
                + self._availability_by_hour[day]
            )

        self._base_start_minutes -= slot_count * self._slot_minutes

    def get_name(self) -> str:
        """Return the person's name."""
        return self._name

    def get_staff_type(self) -> str:
        """Return the person's staff type/role."""
        return self._staff_type

    def set_staff_type(self, staff_type: str) -> None:
        """Set the person's staff type/role."""
        self._staff_type = staff_type

    def get_availability(self, day: str) -> List[int]:
        """Return the availability array for a specific day."""
        return self._availability_by_hour.get(day, [])

    def get_work_capacity_ratio(self) -> float:
        """
        Return the ratio of filled work slots to total available
        school-time slots.
        """
        total_filled_slots = 0
        total_free_slots = 0

        for day in self._availability_by_hour:
            slots = self._availability_by_hour[day]
            total_filled_slots += slots.count(1)
            total_free_slots += slots.count(0)

        total_slots = total_filled_slots + total_free_slots

        if total_slots == 0:
            return 0.0

        return float(total_filled_slots) / total_slots

    def get_hours_worked(self) -> float:
        """Return the total number of hours worked."""
        return sum(self.get_hours_worked_by_day().values())

    def get_hours_worked_by_day(self):
        """Return a dictionary of hours worked per day."""
        hours = {}

        for day, slots in self._availability_by_hour.items():
            hours[day] = (
                slots.count(1) * self._slot_minutes / 60
            )

        return hours

    def get_rest_periods_by_day(self):
        """Return a dictionary of rest periods per day."""
        rest_periods = {}

        for day, slots in self._availability_by_hour.items():
            rest_periods[day] = (
                slots.count(0) * self._slot_minutes / 60
            )

        return rest_periods

    def get_hours_in_school(self) -> float:
        """
        Return the total hours the person is present in school,
        including both working and free slots.
        """
        total_slots_in_school = 0

        for day in self._availability_by_hour:
            slots = self._availability_by_hour[day]

            total_slots_in_school += (
                slots.count(0) + slots.count(1)
            )

        return (
            total_slots_in_school * self._slot_minutes / 60
        )

    def set_availability(
        self,
        day: str,
        start_time: str,
        end_time: str,
        status: int
    ) -> None:
        """
        Mark a person's availability status for a time range.

        Args:
            day: Day key used in the schedule.
            start_time: Start time in HHMM format.
            end_time: End time in HHMM format.
            status:
                -1 = unavailable
                 0 = free
                 1 = filled/working
        """
        if day not in self._availability_by_hour:
            self._availability_by_hour[day] = (
                [-1] * self._get_number_of_slots()
            )

        start_index = self.time_to_index(
            start_time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        end_index = self.time_to_index(
            end_time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        # If the requested start is before the current grid,
        # expand the grid to accommodate it.
        if start_index < 0:
            self._prepend_slots(-start_index)

            start_index = self.time_to_index(
                start_time,
                self._base_start_minutes,
                self._slot_minutes,
            )

            end_index = self.time_to_index(
                end_time,
                self._base_start_minutes,
                self._slot_minutes,
            )

        # If the requested end is beyond the current grid,
        # extend the grid with unavailable slots.
        if end_index > len(self._availability_by_hour[day]):
            self._availability_by_hour[day].extend(
                [-1] * (
                    end_index
                    - len(self._availability_by_hour[day])
                )
            )

        for i in range(start_index, end_index):
            self._availability_by_hour[day][i] = status

    def check_availability(
        self,
        day: str,
        start_time: str,
        end_time: str
    ) -> bool:
        """Check whether the person is free for a given time range."""
        start_index = self.time_to_index(
            start_time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        end_index = self.time_to_index(
            end_time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        if day not in self._availability_by_hour:
            return False

        if start_index < 0 or end_index > len(
            self._availability_by_hour[day]
        ):
            return False

        return (
            np.asarray(
                self._availability_by_hour[day][start_index:end_index]
            ) == 0
        ).all()

    def add_duty(
        self,
        day,
        duty_name=None,
        duty_info: dict = None
    ):
        """
        Add a duty assignment to a person.

        If only a day is provided, maintain the backwards-compatible
        behaviour of recording the assigned day.
        """
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
        """Return all duties assigned on a particular day."""
        return self._duties_by_day.get(day, [])

    def get_activity_at(self, day: str, time: str) -> str:
        """
        Return what the person is doing at a given time.

        Returns:
            Duty name, "Rest", or an empty string.
        """
        slot = self.time_to_index(
            time,
            self._base_start_minutes,
            self._slot_minutes,
        )

        if day not in self._availability_by_hour:
            return ""

        availability = self._availability_by_hour[day]

        if slot < 0 or slot >= len(availability):
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
        Build a schedule using the configured slot duration.

        Returns:
            list[dict]: Schedule entries containing start, end,
            and activity.
        """
        schedule = []

        availability = self.get_availability(day)

        for slot, status in enumerate(availability):
            start_minutes = (
                self._base_start_minutes
                + slot * self._slot_minutes
            )

            end_minutes = start_minutes + self._slot_minutes

            start_hour, start_minute = divmod(
                start_minutes,
                60
            )

            end_hour, end_minute = divmod(
                end_minutes,
                60
            )

            start_time = (
                f"{start_hour:02}{start_minute:02}"
            )

            end_time = (
                f"{end_hour:02}{end_minute:02}"
            )

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

            schedule.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "activity": activity,
                }
            )

        return schedule