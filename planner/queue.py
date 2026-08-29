import math
from random import shuffle
from typing import List, Optional, Callable

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

    def add_to_queue(
        self,
        staff_member: str,
        day: str,
        start_time: str,
        end_time: str,
        status: int,
        staff_type: str = None,
        expected_capacity: float = None,
    ) -> None:
        for index, entry in enumerate(self._queue):
            if entry.get_name() == staff_member:
                self._queue[index].set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
                if staff_type and not self._queue[index].get_staff_type():
                    self._queue[index].set_staff_type(staff_type)
                if expected_capacity is not None:
                    self._queue[index].set_expected_capacity(expected_capacity, day=day)
                return
        person_to_add = Person(
            staff_member,
            staff_type=staff_type,
            expected_capacity=1.0 if expected_capacity is None else expected_capacity,
        )
        person_to_add.set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
        if expected_capacity is not None:
            person_to_add.set_expected_capacity(expected_capacity, day=day)
        self._queue.append(person_to_add)

    def select_available_person(
        self,
        day: str,
        start_time: str,
        end_time: str,
        person_filter: Optional[Callable[[Person], bool]] = None
    ) -> Optional[Person]:
        """Select the first available person who matches optional filter criteria.

        The selected person is marked as assigned for the given time range and moved
        to the back of the queue to preserve fairness.

        Args:
            day (str): Normalized day identifier for the assignment.
            start_time (str): Assignment start time in HHMM format.
            end_time (str): Assignment end time in HHMM format.
            person_filter (callable, optional): A predicate that must return True for a valid person.

        Returns:
            Optional[Person]: The selected Person instance, or None when no match is found.
        """
        selected_index = None
        selected_ratio = None

        # Prefer the available person with the lowest work-to-expected-capacity ratio.
        # Queue order is used only as a tie-breaker so existing round-robin behaviour
        # is preserved when capacities (and current loads) are equal.
        for idx, person in enumerate(self._queue):
            # Check if the person is available for the given day and time range. If not, skip to the next person.
            if not person.check_availability(day=day, start_time=start_time, end_time=end_time):
                continue
            # If a person_filter is provided, apply it to the person. If the filter returns False, skip this person.
            if person_filter is not None and not person_filter(person):
                continue
            ratio = self._get_work_capacity_ratio(person, day)
            if selected_index is None or ratio < selected_ratio:
                selected_index = idx
                selected_ratio = ratio

        if selected_index is not None:
            self._queue[selected_index].set_availability(
                day=day,
                start_time=start_time,
                end_time=end_time,
                status=1
            )
            selected_person = self._queue.pop(selected_index)
            self._queue.append(selected_person)
            return selected_person

        return None

    def get_list(self) -> List[Person]:
        return self._queue

    def shuffle(self) -> None:
        shuffle(self._queue)

    @staticmethod
    def _get_work_capacity_ratio(person: Person, day: str = None) -> float:
        return person.get_work_capacity_ratio(day)

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
