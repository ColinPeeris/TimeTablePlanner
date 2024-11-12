import copy
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from random import shuffle


class Person:
    """
    A class representing a person (e.g., a teacher or temp) involved in duty scheduling.

    Attributes:
        _name (str): The name of the person.
        _capacity (int): The number of days the person is available to work. (Unused in this code but can be extended.)
        _availability (list): A list of days the person is available for duties.
        _days_assigned (list): A list of days the person has been assigned duties.

    Methods:
        __init__: Initializes a new Person object with the given name.
        get_name: Returns the name of the person.
        get_availability: Returns the list of days the person is available.
        set_availability: Adds a day to the person's availability.
        check_availability: Checks if the person is available on a given day.
        add_duty: Adds a duty assignment for the person on a specified day.
        get_duty_list: Returns the list of days the person has been assigned duties.
        get_number_of_duties_taken: Returns the total number of duties assigned to the person.
    """

    def __init__(self, name):
        """
        Initializes a Person object with a given name and sets initial values for capacity,
        availability, and assigned days.

        Args:
            name (str): The name of the person.
        """
        self._name = name
        self._capacity = 0
        self._availability = []  # @todo: remove
        self._days_assigned = []
        self._availability_by_hour = {}
        # -1: Not in school
        # 0: available
        # 1: on duty

    @staticmethod
    def time_to_index(time: str) -> int:
        hours, minutes = divmod(int(time), 100)
        # 9 AM is the reference time, so we subtract 9 * 60 (9 AM = 540 minutes)
        return (hours * 60 + minutes - 540) // 30

    def get_name(self) -> str:
        """
        Returns the name of the person.

        Returns:
            str: The name of the person.
        """
        return self._name

    def get_availability(self, day) -> List[int]:
        """
        Returns the list of days the person is available to take on duties.

        Returns:
            list: A list of days the person is available for duty.
        """
        return self._availability_by_hour[day]
        return self._availability

    def get_work_capacity_ratio(self):
        total_filled_slots = 0
        total_free_slots = 0
        for day in self._availability_by_hour:
            total_filled_slots = total_filled_slots + self._availability_by_hour[day].count(1)
            total_free_slots = total_free_slots + self._availability_by_hour[day].count(0)
        work_to_capacity_ratio = float(total_filled_slots) / float(total_filled_slots + total_free_slots)
        return work_to_capacity_ratio

    '''def set_availability(self, day: str) -> None:
        """
        Adds a day to the person's availability list.

        Args:
            day (str): The day the person becomes available for duty.
        """
        self._availability.append(day)'''

    def set_availability(self, day: str, start_time: str, end_time: str, status: str) -> None:
        """
        Adds a day to the person's availability list.

        Args:
            day (str): The day the person becomes available for duty.
        """
        if day not in self._availability_by_hour:
            self._availability_by_hour[day] = [-1] * 18
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)

        # Update the schedule from start_index to end_index with the given status
        for i in range(start_index, end_index):
            self._availability_by_hour[day][i] = status

    def check_availability_by_day(self, day: str) -> bool:  # @todo: remove
        """
        Checks if the person is available on a specific day.

        Args:
            day (str): The day to check availability for.

        Returns:
            bool: True if the person is available on the specified day, False otherwise.
        """
        return True if day in self._availability else False

    def check_availability(self, day: str, start_time: str, end_time: str) -> bool:
        """
        Checks if the person is available between start_time and end_time.

        Args:
            start_time: The start time in 'HHMM' format (24-hour).
            end_time: The end time in 'HHMM' format (24-hour).

        Returns:
            bool: True if the person is available for the entire time range, False otherwise.
        """
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)
        if day not in self._availability_by_hour:
            return False
        if (np.asarray(self._availability_by_hour[day][start_index:end_index]) == 0).all():
            return True
        else:
            return False

    def add_duty(self, day: str) -> None:
        """
        Adds a duty assignment to the person's schedule for a specific day.

        Args:
            day (str): The day the person is assigned to a duty.
        """
        self._days_assigned.append(day)

    def get_duty_list(self) -> List[str]:
        """
        Returns the list of days the person has been assigned duties.

        Returns:
            list: A list of days the person has been assigned duties.
        """
        return self._days_assigned

    def get_number_of_duties_taken(self) -> int:
        """
        Returns the total number of duties assigned to the person.

        Returns:
            int: The total number of duties the person has been assigned.
        """
        return len(self._days_assigned)


class Queue:
    """
    A queue to manage and schedule people (e.g., teachers or temps) for duties based on availability.

    Attributes:
        _queue (list): A list that stores `Person` objects representing the people available for duty.

    Methods:
        __init__: Initializes a new Queue object.
        _create_queue: Creates a list of Person objects from a given list of names.
        add_to_queue: Adds a `Person` to the queue, updating their availability.
        select_available_person: Selects and removes a person from the queue who is available for a given day.
        get_list: Returns the current list of people in the queue.
        shuffle: Shuffles the people in the queue in random order.
        _get_work_capacity_ratio: Calculates the ratio of duties taken to availability for a given person.
        find_std_deviation: Calculates the standard deviation of the work-to-capacity ratio for all people in the queue.
    """

    def __init__(self):
        """
        Initializes a new Queue object with an empty list of people.
        """
        self._queue = []

    @staticmethod
    def _create_queue(list_of_persons: List) -> List[Person]:
        """
        Creates a list of Person objects from a given list of names.

        Args:
            list_of_persons (List[str]): A list of names representing people to be added to the queue.

        Returns:
            List[Person]: A list of `Person` objects corresponding to the names in the input list.
        """
        queue = []
        for name in list_of_persons:
            person = Person(name=name)
            queue.append(person)
        return queue

    def add_to_queue(self, staff_member: str, day: str, start_time: str, end_time: str, status: int) -> None:
        """
        Adds a `Person` object to the queue, updating their availability if they are already in the queue.

        Args:
            person (Person): The `Person` object to add to the queue.
        """
        # Iterate through the queue to find if the person already exists in the queue
        for index, entry in enumerate(self._queue):
            # Check if the current entry in the queue has the same name as the given person
            if entry.get_name() == staff_member:
                # If the person already exists, update their availability in the queue
                self._queue[index].set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
                return
        # If the person was not found in the queue, add them to the queue
        person_to_add = Person(staff_member)
        person_to_add.set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
        self._queue.append(person_to_add)

    # @todo: remove
    def select_available_person_by_day(self, day: str) -> Optional[Person]:
        """
        Selects an available person from the queue for a specific day. The person is removed from the queue
        once assigned to a duty and is added back to the queue with the duty recorded.

        Args:
            day (str): The day to assign a person to a duty.

        Returns:
            Person or None: The selected `Person` object if available, or None if no one is available.
        """
        selected_index = None
        for idx, person in enumerate(self._queue):
            if person.check_availability(day=day):
                # print(person.get_name())
                # print(person.get_duty_list())
                if day not in person.get_duty_list():
                    selected_index = idx
                    break
        if selected_index is not None:
            selected_person = self._queue.pop(selected_index)
            selected_person.add_duty(day)
            self._queue.append(selected_person)
            # print(f'selected: {selected_person.get_name()}')
            return selected_person
        '''else:
            print('Nobody available')'''
        return None

    def select_available_person(self, day, start_time, end_time) -> Optional[Person]:
        selected_index = None
        for idx, person in enumerate(self._queue):
            if person.check_availability(day=day, start_time=start_time, end_time=end_time):
                selected_index = idx
                break
        if selected_index is not None:
            self._queue[selected_index].set_availability(day=day, start_time=start_time, end_time=end_time, status=1)
            return self._queue[selected_index]

        # no one from the queue is available
        return None

    def get_list(self) -> List[Person]:
        """
        Returns the current list of people in the queue.

        Returns:
            List[Person]: The list of `Person` objects currently in the queue.
        """
        return self._queue

    def shuffle(self) -> None:
        """
        Shuffles the order of people in the queue randomly.
        """
        shuffle(self._queue)

    @staticmethod
    def _get_work_capacity_ratio(person: Person) -> float:
        """
        Calculates the work-to-capacity ratio for a given person.

        Args:
            person (Person): The `Person` object to calculate the ratio for.

        Returns:
            float: The ratio of the number of duties taken to the person's availability.
        """
        return person.get_work_capacity_ratio()

    def find_std_deviation(self) -> float:
        """
        Calculates the standard deviation of the work-to-capacity ratio for all people in the queue.

        Returns:
            float: The standard deviation of the work-to-capacity ratios of all people in the queue.
        """
        if len(self._queue) == 0:
            return
        mean_work_to_capacity_ratio = 0
        for person in self._queue:
            mean_work_to_capacity_ratio = mean_work_to_capacity_ratio + self._get_work_capacity_ratio(person)
        mean_work_to_capacity_ratio = mean_work_to_capacity_ratio / float(len(self._queue))

        sum_of_x = 0
        for person in self._queue:
            x = (self._get_work_capacity_ratio(person) - mean_work_to_capacity_ratio)
            sum_of_x = x * x
        std_deviation = sum_of_x / len(self._queue)
        std_deviation = math.sqrt(std_deviation)
        return std_deviation


class DutyRoster:
    def __init__(self):
        self._duty_roster = {}

    @staticmethod
    def calculate_duration(start_time: str, end_time: str) -> float:
        if isinstance(start_time, int):
            start_time = str(start_time)
        if isinstance(end_time, int):
            end_time = str(end_time)
        # Ensure both times have 4 digits by padding with leading zeros if necessary
        start_time = start_time.zfill(4)
        end_time = end_time.zfill(4)

        # Extract hours and minutes from start_time and end_time
        start_hour = int(start_time[:2])
        start_minute = int(start_time[2:])

        end_hour = int(end_time[:2])
        end_minute = int(end_time[2:])

        # Convert both start_time and end_time to minutes from midnight
        start_total_minutes = start_hour * 60 + start_minute
        end_total_minutes = end_hour * 60 + end_minute

        # If the end time is earlier than the start time, it means we passed midnight
        if end_total_minutes < start_total_minutes:
            end_total_minutes += 24 * 60  # Add 24 hours in minutes (1440 minutes)

        # Calculate the duration in minutes
        duration_minutes = end_total_minutes - start_total_minutes

        # Convert the duration to hours
        return duration_minutes / 60

    def add_day(self, day):
        if day not in self._duty_roster:
            self._duty_roster[day] = {}

    def add_duty(self, activity, session, start_time, end_time, min_requirement, ideal_case):
        for day in self._duty_roster:
            self._duty_roster[day][activity] = {
                'session': session,
                'start_time': start_time,
                'end_time': end_time,
                'duration': self.calculate_duration(start_time=start_time, end_time=end_time),
                'min_requirement': min_requirement,
                'ideal_case': ideal_case,
                'assignees': []
            }

    def get_duty_roster(self):
        # @todo: make this better
        return self._duty_roster


class Scheduler:
    """
    The Scheduler class is responsible for assigning teachers and temporary staff (temps) to various slots
    based on their availability, and optimizing the distribution of duties. It uses data from an Excel file
    that contains the availability of teachers and temps in AM and PM time slots. The scheduler optimizes
    the distribution using a shuffling and standard deviation minimization approach over 100 iterations.

    Attributes:
        None

    Methods:
        __init__: Initializes the Scheduler by reading data from an Excel file, processing the availability
                  of teachers and temps, and performing duty assignment and optimization.
        _write_results_to_excel: Saves the final duty assignments and work distribution to an Excel file.
        _get_by_am_pm: Helper method to extract availability data for AM and PM time slots from a DataFrame.
        _get_data_by_sheet: Combines the AM and PM availability data for teachers and temps from Excel sheets.
        _get_data_from_excel: Reads availability data for teachers and temps from an Excel file.
        _create_duties_per_slot: Creates a duty list for a specific time slot, assigning available teachers and
                                 temps based on predefined roles and conditions (e.g., PreN, N, K for AM, N, K for PM).
    """

    def __init__(self):
        """
        Initializes the Scheduler by loading teacher and temp availability data from an Excel file
        ('AvailabilityList.xlsx'), processes the availability, and attempts to optimize the duty assignment.

        Steps:
            - Loads the availability data for teachers and temps.
            - Creates queues for teachers and temps for each slot.
            - Performs 100 iterations of shuffling the queues and assigns duties, trying to minimize the
              standard deviation of duty assignments.
            - Outputs the best result after 100 iterations to an Excel file.
        """

        teachers_am_list, teachers_pm_list, temps_am_list, temps_pm_list = self._get_staff_availability(
            'AvailabilityList.xlsx')

        self._duty_roster = DutyRoster()

        for slot in teachers_am_list:
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"
            self._duty_roster.add_day(day)

        self._get_duties_list_from_excel('DutiesBreakdown.xlsx')

        teacher_list = Queue()
        temp_list = Queue()

        # Process all the slots
        self._add_to_queue_for_slot(teacher_list, teachers_am_list, '0900', '1400')
        self._add_to_queue_for_slot(teacher_list, teachers_pm_list, '1400', '1800')
        self._add_to_queue_for_slot(temp_list, temps_am_list, '0900', '1400')
        self._add_to_queue_for_slot(temp_list, temps_pm_list, '1400', '1800')

        # iterate 100 times until the distribution is most fair...
        min_std_deviation = 99999
        success_count = 0
        finalized_teacher_list = None
        finalized_temp_list = None
        final_duty_list = None
        final_roster = None

        for i in range(0, 100):
            # make copies of teacher and temp queues
            _teacher_list = copy.deepcopy(teacher_list)
            _temp_list = copy.deepcopy(temp_list)

            duty_roster = copy.deepcopy(self._duty_roster.get_duty_roster())
            for day in duty_roster:
                # shuffle queue at the start of the day
                _teacher_list.shuffle()
                _temp_list.shuffle()
                for duty in duty_roster[day]:
                    print(f'setting duties for {duty}...')
                    #for i in range(0, int(duty_roster[day][duty]['ideal_case'])):
                    for i in range(0, int(duty_roster[day][duty]['min_requirement'])):
                        selected_teacher = None
                        selected_temp = None
                        selected_teacher = _teacher_list.select_available_person(
                            day=day,
                            start_time=duty_roster[day][duty]['start_time'],
                            end_time=duty_roster[day][duty]['end_time']
                        )

                        # If no teacher is found, try to assign a temp
                        if selected_teacher is not None:
                            duty_roster[day][duty]['assignees'].append(selected_teacher)
                            print(f'Selected {selected_teacher.get_name()} for {day}')
                        else:
                            selected_temp = _temp_list.select_available_person(
                                day=day,
                                start_time=duty_roster[day][duty]['start_time'],
                                end_time=duty_roster[day][duty]['end_time']
                            )
                            if selected_temp is not None:
                                duty_roster[day][duty]['assignees'].append(selected_temp)
                                print(f'Selected {selected_temp.get_name()} for {day}')

                        # If neither teacher nor temp is found and it's not the last iteration, raise an error
                        if (selected_teacher is None) and (selected_temp is None) and (
                                i + 1 < int(duty_roster[day][duty]['min_requirement'])):
                            assert False, f"Error: Could not find enough staff for {day} " \
                                          f"from {duty_roster[day][duty]['start_time']} to {duty_roster[day][duty]['end_time']}. " \
                                          f"Minimum requirement not met."

            sum_of_std_deviation = _teacher_list.find_std_deviation() + _temp_list.find_std_deviation()
            if sum_of_std_deviation < min_std_deviation:
                min_std_deviation = sum_of_std_deviation
                finalized_teacher_list = copy.deepcopy(_teacher_list)
                finalized_temp_list = copy.deepcopy(_temp_list)
                assert duty_roster is not None
                final_roster = duty_roster

            success_count += 1

        print(f'Number of successful runs: {success_count}')

        if (finalized_temp_list is not None) and (finalized_teacher_list is not None):
            print('------------------------------------')
            print('Print Best Results:')
            for person in finalized_temp_list.get_list():
                print(f'Person: {person.get_name()}, Work-To-Capacity: {person.get_work_capacity_ratio()}')
            for person in finalized_teacher_list.get_list():
                print(f'Person: {person.get_name()}, Work-To-Capacity: {person.get_work_capacity_ratio()}')
            self._write_roster_to_excel(final_roster)

    @staticmethod
    def _add_to_queue_for_slot(queue, slot_list, start_time, end_time):
        for slot in slot_list:
            # Create the day string
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"

            # Clean the staff members list by removing NaNs and stripping whitespace
            staff_members = [staff_member.strip() for staff_member in slot[2:] if pd.notna(staff_member)]

            # Add each staff member to the queue
            for staff_member in staff_members:
                queue.add_to_queue(staff_member=staff_member, day=day, start_time=start_time, end_time=end_time,
                                   status=0)

    @staticmethod
    def _write_roster_to_excel(roster: dict) -> None:
        duties = []
        teachers_by_day = {}  # Dictionary to store teacher lists by day

        for day in roster:
            # Initialize a list of duties for each day in the dictionary
            teachers_by_day[day] = []

            for duty in roster[day]:
                duties.append(duty)  # Add the duty to the duties list

                # Extract assignees for the current duty (duty['assignees'])
                assignees = roster[day][duty]['assignees']

                # Create a list of teachers for the current duty
                teachers_for_duty = []

                # Loop through assignees and add their names to the current duty's teacher list
                for idx, assignee in enumerate(assignees):
                    teachers_for_duty.append(assignee.get_name())

                # If there are fewer assignees than the max, append 'NA' or placeholder
                for idx in range(len(assignees), 6):  # Assuming max 6 assignees per duty
                    teachers_for_duty.append('NA')  # Placeholder for missing teachers

                # Add the current duty's teachers to the list for that day
                teachers_by_day[day].append((duty, teachers_for_duty))  # Storing duty name along with teachers

        # Print the structured teachers_by_day dictionary for debugging
        print(teachers_by_day)

        # Prepare data for Excel with duty as the second column
        data_for_excel = []

        # Loop through each day in the teachers_by_day dictionary
        for day, duties in teachers_by_day.items():
            # For each duty on the given day, prepend the day and then duty to the list of teachers
            for duty, duty_teachers in duties:
                # Create a row: first the day, then the duty, followed by the teachers
                row = [day, duty] + duty_teachers  # Add day, duty, and teacher names
                data_for_excel.append(row)

        # Now, create a DataFrame from the list of rows
        columns = ['Day', 'Duty', 'Teacher 1', 'Teacher 2', 'Teacher 3', 'Teacher 4', 'Teacher 5', 'Teacher 6']

        df = pd.DataFrame(data_for_excel, columns=columns)

        # Write the DataFrame to an Excel file
        file_name = 'teacher_schedule_with_duties.xlsx'
        df.to_excel(file_name, index=False)

        print(f"Data has been written to {file_name}")

    @staticmethod
    def _get_staff_availability(file_name) -> Tuple[List, List, List, List]:
        df_teachers_am = pd.read_excel(file_name, sheet_name='Teachers_AM')
        df_teachers_pm = pd.read_excel(file_name, sheet_name='Teachers_PM')
        df_temps_am = pd.read_excel(file_name, sheet_name='Temps_AM')
        df_temps_pm = pd.read_excel(file_name, sheet_name='Temps_PM')

        teachers_am_list = df_teachers_am.values.tolist()
        teachers_pm_list = df_teachers_pm.values.tolist()
        temps_am_list = df_temps_am.values.tolist()
        temps_pm_list = df_temps_pm.values.tolist()

        return teachers_am_list, teachers_pm_list, temps_am_list, temps_pm_list

    def _get_duties_list_from_excel(self, file_name):
        dataframe = pd.read_excel(file_name)
        for activity, session, start_time, end_time, min_requirement, ideal_case in \
                zip(dataframe['Activity'], dataframe['Session'], dataframe['Start Time'], dataframe['End Time'], \
                dataframe['Minimum Requirement'], dataframe['Ideal Case']):
            self._duty_roster.add_duty(
                activity=activity,
                session=session,
                start_time=start_time,
                end_time=end_time,
                min_requirement=min_requirement,
                ideal_case=ideal_case
            )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scheduler = Scheduler()

# To run it, run the following:
# python .\main.py
