from typing import List

import numpy as np


class Person:
    """
    A class representing a person (e.g., a teacher or temp) who can be assigned duties based on availability.

    Attributes:
        _name (str): The name of the person.
        _availability_by_hour (dict): The person's availability by day and 30-minute slot.
        _days_assigned (list): Days when the person has duties assigned.
    """

    def __init__(self, name: str):
        self._name = name
        self._availability_by_hour = {}
        self._days_assigned = []

    @staticmethod
    def time_to_index(time: str) -> int:
        hours, minutes = divmod(int(time), 100)
        return (hours * 60 + minutes - 540) // 30

    def get_name(self) -> str:
        return self._name

    def get_availability(self, day: str) -> List[int]:
        return self._availability_by_hour.get(day, [])

    def get_work_capacity_ratio(self) -> float:
        total_filled_slots = 0
        total_free_slots = 0
        for day in self._availability_by_hour:
            total_filled_slots += self._availability_by_hour[day].count(1)
            total_free_slots += self._availability_by_hour[day].count(0)
        return float(total_filled_slots) / (total_filled_slots + total_free_slots) \
            if (total_filled_slots + total_free_slots) > 0 else 0.0

    def get_hours_worked(self) -> float:
        total_filled_slots = 0
        for day in self._availability_by_hour:
            total_filled_slots += self._availability_by_hour[day].count(1)
        return float(total_filled_slots) / 2

    def get_hours_in_school(self) -> float:
        total_slots_in_school = 0
        for day in self._availability_by_hour:
            total_slots_in_school += self._availability_by_hour[day].count(0) + self._availability_by_hour[day].count(1)
        return float(total_slots_in_school) / 2

    def set_availability(self, day: str, start_time: str, end_time: str, status: int) -> None:
        if day not in self._availability_by_hour:
            self._availability_by_hour[day] = [-1] * 18
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)
        for i in range(start_index, end_index):
            self._availability_by_hour[day][i] = status

    def check_availability(self, day: str, start_time: str, end_time: str) -> bool:
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)
        if day not in self._availability_by_hour:
            return False
        return (np.asarray(self._availability_by_hour[day][start_index:end_index]) == 0).all()

    def add_duty(self, day: str) -> None:
        self._days_assigned.append(day)
