import math
from random import shuffle
from typing import List, Optional

from .person import Person


class Queue:
    """
    A class to manage a queue of people for duty assignments based on availability.
    """

    def __init__(self):
        self._queue = []

    @staticmethod
    def _create_queue(list_of_persons: List[str]) -> List[Person]:
        queue = []
        for name in list_of_persons:
            queue.append(Person(name=name))
        return queue

    def add_to_queue(self, staff_member: str, day: str, start_time: str, end_time: str, status: int) -> None:
        for index, entry in enumerate(self._queue):
            if entry.get_name() == staff_member:
                self._queue[index].set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
                return
        person_to_add = Person(staff_member)
        person_to_add.set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
        self._queue.append(person_to_add)

    def select_available_person(self, day: str, start_time: str, end_time: str) -> Optional[Person]:
        selected_index = None
        for idx, person in enumerate(self._queue):
            if person.check_availability(day=day, start_time=start_time, end_time=end_time):
                selected_index = idx
                break
        if selected_index is not None:
            self._queue[selected_index].set_availability(day=day, start_time=start_time, end_time=end_time, status=1)
            selected_person = self._queue.pop(selected_index)
            self._queue.append(selected_person)
            return selected_person
        return None

    def get_list(self) -> List[Person]:
        return self._queue

    def shuffle(self) -> None:
        shuffle(self._queue)

    @staticmethod
    def _get_work_capacity_ratio(person: Person) -> float:
        return person.get_work_capacity_ratio()

    def find_std_deviation(self) -> float:
        if len(self._queue) == 0:
            return 0.0
        mean_work_to_capacity_ratio = 0.0
        for person in self._queue:
            mean_work_to_capacity_ratio += self._get_work_capacity_ratio(person)
        mean_work_to_capacity_ratio /= len(self._queue)
        sum_of_x_squared = 0
        for person in self._queue:
            x = self._get_work_capacity_ratio(person) - mean_work_to_capacity_ratio
            sum_of_x_squared += x * x
        return math.sqrt(sum_of_x_squared / len(self._queue))
