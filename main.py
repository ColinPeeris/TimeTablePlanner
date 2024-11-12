import copy
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from random import shuffle


class Person:
    """
    A class representing a person (e.g., a teacher or temp) who can be assigned duties based on availability.

    This class manages the availability of a person for scheduled duties and keeps track of the days they have been assigned.

    Attributes:
        _name (str): The name of the person.
        _availability_by_hour (dict): A dictionary mapping days to lists representing the person's availability
                                      for each 30-minute time slot (from 9:00 AM onward).
                                      Each list contains values indicating availability:
                                        - -1: Not in school
                                        - 0: Available
                                        - 1: On duty
        _days_assigned (list): A list of days the person has been assigned duties.

    Methods:
        __init__: Initializes a new Person object with the given name and sets up availability and duty tracking.
        get_name: Returns the name of the person.
        get_availability: Returns the availability of the person for a specific day.
        get_work_capacity_ratio: Calculates and returns the ratio of duties assigned to the total availability.
        set_availability: Sets the availability status (0 for available, 1 for on duty) for a given time range on a specified day.
        check_availability: Checks if the person is available for a specific time range on a given day.
        add_duty: Adds a day to the list of days the person has been assigned a duty.
    """

    def __init__(self, name: str):
        """
        Initializes a Person object with the given name and sets up initial availability tracking.

        Args:
            name (str): The name of the person.
        """
        self._name = name
        self._availability_by_hour = {}
        self._days_assigned = []

    @staticmethod
    def time_to_index(time: str) -> int:
        """
        Converts a time string (HHMM) to an index representing the corresponding 30-minute slot from 9:00 AM.

        Args:
            time (str): A time string in 'HHMM' format (24-hour).

        Returns:
            int: An index representing the 30-minute slot, where 9:00 AM corresponds to index 0.
        """
        hours, minutes = divmod(int(time), 100)
        return (hours * 60 + minutes - 540) // 30

    def get_name(self) -> str:
        """
        Returns the name of the person.

        Returns:
            str: The name of the person.
        """
        return self._name

    def get_availability(self, day: str) -> List[int]:
        """
        Returns the availability of the person for a specific day.

        Args:
            day (str): The day to check availability.

        Returns:
            list: A list of 18 integers representing the availability for the day,
                  where each integer corresponds to a 30-minute time slot (from 9:00 AM onward).
                  Values are:
                    - -1: Not in school
                    - 0: Available
                    - 1: On duty
        """
        return self._availability_by_hour.get(day, [])

    def get_work_capacity_ratio(self) -> float:
        """
        Calculates the ratio of duties assigned to the total availability of the person.

        The ratio is the number of filled duty slots divided by the total number of available slots across all days.

        Returns:
            float: The work-to-capacity ratio, indicating how much of the person's availability has been utilized.
        """
        total_filled_slots = 0
        total_free_slots = 0
        for day in self._availability_by_hour:
            total_filled_slots += self._availability_by_hour[day].count(1)
            total_free_slots += self._availability_by_hour[day].count(0)
        return float(total_filled_slots) / (total_filled_slots + total_free_slots) if (total_filled_slots + total_free_slots) > 0 else 0.0

    def set_availability(self, day: str, start_time: str, end_time: str, status: int) -> None:
        """
        Sets the availability status for a given time range on a specified day.

        The status can be 0 for available or 1 for on duty. The time range is represented by start_time and end_time
        in 'HHMM' format.

        Args:
            day (str): The day for which to set availability.
            start_time (str): The start time of the availability in 'HHMM' format (24-hour).
            end_time (str): The end time of the availability in 'HHMM' format (24-hour).
            status (int): The status to set for the time range (0 for available, 1 for on duty).
        """
        if day not in self._availability_by_hour:
            self._availability_by_hour[day] = [-1] * 18  # Initialize the day's availability with "Not in school"
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)

        # Update the schedule from start_index to end_index with the given status
        for i in range(start_index, end_index):
            self._availability_by_hour[day][i] = status

    def check_availability(self, day: str, start_time: str, end_time: str) -> bool:
        """
        Checks if the person is available for a specific time range on a given day.

        Args:
            day (str): The day to check availability.
            start_time (str): The start time in 'HHMM' format (24-hour).
            end_time (str): The end time in 'HHMM' format (24-hour).

        Returns:
            bool: True if the person is available for the entire time range, False otherwise.
        """
        start_index = self.time_to_index(start_time)
        end_index = self.time_to_index(end_time)
        if day not in self._availability_by_hour:
            return False
        # Check if the person is available (status 0) for the entire time range
        return (np.asarray(self._availability_by_hour[day][start_index:end_index]) == 0).all()

    def add_duty(self, day: str) -> None:
        """
        Adds a duty assignment to the person's schedule for a specific day.

        Args:
            day (str): The day the person is assigned to a duty.
        """
        self._days_assigned.append(day)


class Queue:
    """
    A class to manage a queue of people (e.g., teachers or temps) for duty assignments based on their availability.

    This class allows adding people to the queue, checking availability for duty assignments,
    and optimizing the distribution of duties by shuffling or calculating workload distribution metrics.

    Attributes:
        _queue (list): A list that stores `Person` objects representing individuals available for duties.

    Methods:
        __init__: Initializes an empty queue.
        _create_queue: Creates a list of `Person` objects from a provided list of names.
        add_to_queue: Adds a person to the queue or updates their availability if they are already present.
        select_available_person: Selects and removes a person from the queue who is available for a given day and time.
        get_list: Returns the current list of `Person` objects in the queue.
        shuffle: Randomly shuffles the order of people in the queue.
        _get_work_capacity_ratio: Calculates the work-to-capacity ratio for a given person.
        find_std_deviation: Calculates the standard deviation of the work-to-capacity ratios for all people in the queue.
    """

    def __init__(self):
        """
        Initializes a new Queue object with an empty list to store available people.
        """
        self._queue = []

    @staticmethod
    def _create_queue(list_of_persons: List) -> List[Person]:
        """
        Creates a list of `Person` objects from a given list of names.

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
        Adds a person to the queue or updates their availability if they are already in the queue.

        Args:
            staff_member (str): The name of the staff member to add or update.
            day (str): The day for which the availability is being set.
            start_time (str): The start time for the duty.
            end_time (str): The end time for the duty.
            status (int): The availability status (e.g., 0 for unavailable, 1 for available).
        """
        # Check if the person is already in the queue
        for index, entry in enumerate(self._queue):
            if entry.get_name() == staff_member:
                # Update the person's availability
                self._queue[index].set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
                return
        # If the person is not in the queue, create a new Person object and add it
        person_to_add = Person(staff_member)
        person_to_add.set_availability(day=day, start_time=start_time, end_time=end_time, status=status)
        self._queue.append(person_to_add)

    def select_available_person(self, day: str, start_time: str, end_time: str) -> Optional[Person]:
        """
        Selects and removes a person from the queue who is available for the specified day and time range.

        Args:
            day (str): The day to check availability.
            start_time (str): The start time for the duty.
            end_time (str): The end time for the duty.

        Returns:
            Optional[Person]: A `Person` object representing the selected available person, or `None` if no one is available.
        """
        selected_index = None
        for idx, person in enumerate(self._queue):
            if person.check_availability(day=day, start_time=start_time, end_time=end_time):
                selected_index = idx
                break
        if selected_index is not None:
            # Mark the selected person as unavailable for this time slot
            self._queue[selected_index].set_availability(day=day, start_time=start_time, end_time=end_time, status=1)
            return self._queue[selected_index]

        # If no one is available, return None
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
        Randomly shuffles the order of people in the queue.

        This method is useful for randomizing the duty assignments or ensuring fairness in assignments.
        """
        shuffle(self._queue)

    @staticmethod
    def _get_work_capacity_ratio(person: Person) -> float:
        """
        Calculates the work-to-capacity ratio for a given person.

        Args:
            person (Person): The person for whom the work capacity ratio is to be calculated.

        Returns:
            float: The work-to-capacity ratio, representing the balance between duties assigned and availability.
        """
        return person.get_work_capacity_ratio()

    def find_std_deviation(self) -> float:
        """
        Calculates the standard deviation of the work-to-capacity ratios for all people in the queue.

        This is useful for determining how evenly duties are distributed among the available staff.

        Returns:
            float: The standard deviation of the work-to-capacity ratios of all people in the queue.
                   If the queue is empty, returns 0.0.
        """
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

        std_deviation = math.sqrt(sum_of_x_squared / len(self._queue))
        return std_deviation


class DutyRoster:
    """
    The DutyRoster class is responsible for managing the roster of duties for staff members.
    It tracks the duties for each day, calculates duty durations, and maintains a list of assigned staff.

    Attributes:
        _duty_roster (dict): A dictionary where each day maps to its respective duties.

    Methods:
        __init__: Initializes an empty duty roster.
        calculate_duration: Calculates the duration between two given times.
        add_day: Adds a day to the duty roster if it doesn't already exist.
        add_duty: Adds a duty to all days in the duty roster with necessary details.
        get_duty_roster: Returns the current duty roster.
    """

    def __init__(self):
        """
        Initializes an empty duty roster to store duty details for each day.
        """
        self._duty_roster = {}

    @staticmethod
    def calculate_duration(start_time: str, end_time: str) -> float:
        """
        Calculates the duration between the start time and end time in hours.

        Args:
            start_time (str): The start time in the format "HHMM".
            end_time (str): The end time in the format "HHMM".

        Returns:
            float: The duration between start_time and end_time in hours.

        Notes:
            - If the end time is earlier than the start time (indicating the period crosses midnight),
              24 hours are added to the end time for accurate duration calculation.
            - The function assumes start_time and end_time are in the "HHMM" format (e.g., "0930" for 9:30 AM).
        """
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
        """
        Adds a day to the duty roster if it doesn't already exist.

        Args:
            day (str): The day to be added to the roster.

        Notes:
            - This method initializes an empty dictionary of duties for the specified day.
            - If the day already exists in the roster, no action is taken.
        """
        if day not in self._duty_roster:
            self._duty_roster[day] = {}

    def add_duty(self, activity, session, start_time, end_time, min_requirement, ideal_case):
        """
        Adds a duty to all days in the duty roster with the specified details.

        Args:
            activity (str): The name of the duty activity (e.g., "AM Activity").
            session (str): The session during which the duty takes place (e.g., "AM", "PM").
            start_time (str): The start time of the duty in the format "HHMM".
            end_time (str): The end time of the duty in the format "HHMM".
            min_requirement (int): The minimum number of staff members required for this duty.
            ideal_case (int): The ideal number of staff members for this duty.

        Notes:
            - This method adds the duty to each day in the roster, where each day will have an entry with the activity's details.
            - The duty details include the duration (calculated using `calculate_duration`), minimum and ideal staffing requirements,
              and an empty list of assignees.
        """
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
        """
        Returns the current duty roster, which contains the duties assigned to each day.

        Returns:
            dict: A dictionary where the key is the day (e.g., 'Monday') and the value is another dictionary containing
                  duty activities with details such as time, session, requirements, and assignees.

        Notes:
            - The duty roster is a nested dictionary, where each day has its respective duties listed.
        """
        return self._duty_roster


class Scheduler:
    """
    The Scheduler class is responsible for assigning teachers and temporary staff (temps) to various slots
    based on their availability, and optimizing the distribution of duties. The optimization is done by
    shuffling and minimizing the standard deviation over 100 iterations.

    Attributes:
        _duty_roster: An instance of DutyRoster that manages all the duties and assignments.

    Methods:
        __init__: Initializes the scheduler, reads availability data from Excel, assigns duties, and optimizes the schedule.
        _write_results_to_excel: Saves the final duty assignments and work distribution to an Excel file.
        _get_staff_availability: Reads availability data for teachers and temps from an Excel file and returns the lists.
        _get_duties_list_from_excel: Reads duty breakdown from an Excel file and adds it to the duty roster.
        _add_to_queue_for_slot: Helper function to add staff members to the respective queues based on their availability.
    """

    def __init__(self):
        """
        Initializes the Scheduler by reading availability data from an Excel file ('AvailabilityList.xlsx'),
        processes the availability, and attempts to optimize the duty assignment.

        Steps:
            - Loads teacher and temp availability data.
            - Creates duty queues for teachers and temps.
            - Performs 100 iterations of shuffling and assigns duties, trying to minimize the standard deviation.
            - Outputs the best result after 100 iterations to an Excel file.
        """

        # Step 1: Load availability data
        teachers_am_list, teachers_pm_list, temps_am_list, temps_pm_list = self._get_staff_availability(
            'AvailabilityList.xlsx')

        # Step 2: Initialize duty roster
        self._duty_roster = DutyRoster()
        for slot in teachers_am_list:
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"
            self._duty_roster.add_day(day)

        # Step 3: Load duties breakdown from Excel
        self._get_duties_list_from_excel('DutiesBreakdown.xlsx')

        # Step 4: Create queues for teachers and temps
        teacher_list = Queue()
        temp_list = Queue()

        # Process all the slots and add to queues
        self._add_to_queue_for_slot(teacher_list, teachers_am_list, '0900', '1400')
        self._add_to_queue_for_slot(teacher_list, teachers_pm_list, '1400', '1800')
        self._add_to_queue_for_slot(temp_list, temps_am_list, '0900', '1400')
        self._add_to_queue_for_slot(temp_list, temps_pm_list, '1400', '1800')

        # Step 5: Optimize duty assignment through 100 iterations
        best_schedule, finalized_teacher_list, finalized_temp_list = self._optimize_duty_assignment(teacher_list, temp_list)

        # Step 6: Output the final results
        self._write_roster_to_excel(best_schedule, finalized_teacher_list, finalized_temp_list)

    def _add_to_queue_for_slot(self, queue, slot_list, start_time, end_time):
        """
        Adds staff members to a queue based on their availability in each slot.

        Args:
            queue (Queue): The queue to add staff members to.
            slot_list (list): A list of availability slots, where each slot contains day, session, and staff members.
            start_time (str): The start time for the slot.
            end_time (str): The end time for the slot.
        """
        for slot in slot_list:
            # Create the day string by replacing spaces with underscores
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"

            # Clean and filter staff members, remove NaNs and extra spaces
            staff_members = [staff_member.strip() for staff_member in slot[2:] if pd.notna(staff_member)]

            # Add each valid staff member to the queue
            for staff_member in staff_members:
                queue.add_to_queue(staff_member=staff_member, day=day, start_time=start_time, end_time=end_time,
                                   status=0)

    def _optimize_duty_assignment(self, teacher_list, temp_list):
        """
        Optimizes the duty assignment by performing 100 iterations of shuffling the staff queues and
        minimizing the standard deviation of assignments. Returns the best schedule.

        Args:
            teacher_list (Queue): The queue containing available teachers.
            temp_list (Queue): The queue containing available temps.

        Returns:
            dict: The optimized duty roster.
        """
        min_std_deviation = float('inf')
        finalized_teacher_list = None
        finalized_temp_list = None
        final_roster = None

        for _ in range(100):
            # Make copies of the teacher and temp lists to shuffle
            _teacher_list = copy.deepcopy(teacher_list)
            _temp_list = copy.deepcopy(temp_list)

            # Copy of the duty roster for this iteration
            duty_roster = copy.deepcopy(self._duty_roster.get_duty_roster())

            # Shuffle and assign duties for the day
            for day in duty_roster:
                _teacher_list.shuffle()
                _temp_list.shuffle()

                for duty_name, duty_info in duty_roster[day].items():
                    self._assign_staff_to_duty(day, duty_info, _teacher_list, _temp_list, duty_info['min_requirement'], ideal_case=False)

                for duty_name, duty_info in duty_roster[day].items():
                    if duty_info['min_requirement'] < duty_info['ideal_case']:
                        self._assign_staff_to_duty(day, duty_info, _teacher_list, _temp_list, duty_info['ideal_case'] - duty_info['min_requirement'],
                                                   ideal_case=True)


            # Evaluate the quality of this schedule by checking the standard deviation
            sum_of_std_deviation = _teacher_list.find_std_deviation() + _temp_list.find_std_deviation()

            # If this schedule is better (lower std deviation), store it
            if sum_of_std_deviation < min_std_deviation:
                min_std_deviation = sum_of_std_deviation
                finalized_teacher_list = copy.deepcopy(_teacher_list)
                finalized_temp_list = copy.deepcopy(_temp_list)
                final_roster = duty_roster

        return final_roster, finalized_teacher_list, finalized_temp_list

    def _assign_staff_to_duty(self, day, duty_info, teacher_list, temp_list, required_count, ideal_case: bool):
        """
        Assigns staff (teachers or temps) to a duty until the required count is met.

        Args:
            day(str): Which day we're on
            duty_info (dict): The duty information containing start_time, end_time, and assignees.
            teacher_list (Queue): The list of available teachers.
            temp_list (Queue): The list of available temps.
            required_count (int): The number of staff required to be assigned to the duty.
            ideal_case(bool): Boolean of whether assignment is for ideal case or minimum requirement

        Returns:
            None: Staff members are assigned in place to the duty.
        """
        for _ in range(required_count):
            selected_teacher = teacher_list.select_available_person(
                day, duty_info['start_time'], duty_info['end_time']
            )
            selected_temp = None

            # Assign teacher if available, else assign temp
            if selected_teacher:
                duty_info['assignees'].append(selected_teacher)
            else:
                selected_temp = temp_list.select_available_person(
                    day, duty_info['start_time'], duty_info['end_time']
                )
                if selected_temp:
                    duty_info['assignees'].append(selected_temp)

            # Skip following check if we're populating for the ideal case
            if ideal_case:
                return

            # If neither a teacher nor a temp is found, raise an error
            if not selected_teacher and not selected_temp:
                raise ValueError(
                    f"Unable to find sufficient staff for {duty_info['start_time']} to {duty_info['end_time']}"
                )

    @staticmethod
    def _write_roster_to_excel(roster: dict, finalized_teacher_list: Queue, finalized_temp_list: Queue) -> None:
        """
        Writes the duty roster to an Excel file with two sheets:
        1. The first sheet ("Duty Roster") contains the daily duty assignments and the teachers (or temps) assigned to each duty.
        2. The second sheet ("Work Distribution") contains the work-to-capacity ratio for each person (both teachers and temps),
           showing how much work each person has been assigned relative to their availability.

        Args:
            roster (dict): A dictionary representing the duty roster, where each day contains duties with corresponding assignees.
            finalized_teacher_list (Queue): A queue containing the list of teachers with their availability and assigned duties.
            finalized_temp_list (Queue): A queue containing the list of temps with their availability and assigned duties.

        Returns:
            None: The function writes the duty roster and work distribution to an Excel file named 'teacher_schedule_with_duties.xlsx'.

        Example:
            _write_roster_to_excel(roster, finalized_teacher_list, finalized_temp_list)

        The output file will contain:
            - A sheet named "Duty Roster" with the daily duty assignments and the teachers assigned to each duty.
            - A sheet named "Work Distribution" showing the name of each person and their work-to-capacity ratio.
        """
        duties = []
        teachers_by_day = {}

        for day in roster:
            teachers_by_day[day] = []

            for duty in roster[day]:
                assignees = roster[day][duty]['assignees']
                teachers_for_duty = [assignee.get_name() for assignee in assignees] + ['NA'] * (6 - len(assignees))

                teachers_by_day[day].append((duty, teachers_for_duty))

        # Prepare data for the first sheet (duty roster)
        data_for_excel = []

        for day, duties in teachers_by_day.items():
            for duty, duty_teachers in duties:
                row = [day, duty] + duty_teachers
                data_for_excel.append(row)

        # Prepare data for the second sheet (work distribution)
        people = []
        number_of_duties_taken = []
        for person in finalized_temp_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_work_capacity_ratio())
        for person in finalized_teacher_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_work_capacity_ratio())

        work_distribution = pd.DataFrame(
            {'Person': people,
             'Work To Capacity': number_of_duties_taken,
             })

        # Create a DataFrame for the duty roster (first sheet)
        df_roster = pd.DataFrame(data_for_excel,
                                 columns=['Day', 'Duty', 'Teacher 1', 'Teacher 2', 'Teacher 3', 'Teacher 4',
                                          'Teacher 5',
                                          'Teacher 6'])

        # Create a Pandas Excel writer object and write both sheets to the file
        with pd.ExcelWriter('teacher_schedule_with_duties.xlsx', engine='xlsxwriter') as writer:
            # Write the duty roster to the first sheet
            df_roster.to_excel(writer, sheet_name='Duty Roster', index=False)

            # Write the work distribution to the second sheet
            work_distribution.to_excel(writer, sheet_name='Work Distribution', index=False)

        print("Data has been written to teacher_schedule_with_duties.xlsx")

    @staticmethod
    def _get_staff_availability(file_name) -> Tuple[List, List, List, List]:
        """
        Reads the staff availability data for teachers and temps from an Excel file.

        Args:
            file_name (str): The name of the Excel file containing availability data.

        Returns:
            tuple: A tuple of four lists: teachers_am_list, teachers_pm_list, temps_am_list, temps_pm_list.
        """
        df_teachers_am = pd.read_excel(file_name, sheet_name='Teachers_AM')
        df_teachers_pm = pd.read_excel(file_name, sheet_name='Teachers_PM')
        df_temps_am = pd.read_excel(file_name, sheet_name='Temps_AM')
        df_temps_pm = pd.read_excel(file_name, sheet_name='Temps_PM')

        return df_teachers_am.values.tolist(), df_teachers_pm.values.tolist(), df_temps_am.values.tolist(), df_temps_pm.values.tolist()

    def _get_duties_list_from_excel(self, file_name):
        """
        Reads the duties breakdown from an Excel file and adds it to the duty roster.

        Args:
            file_name (str): The name of the Excel file containing duty data.
        """
        dataframe = pd.read_excel(file_name)
        for activity, session, start_time, end_time, min_requirement, ideal_case in zip(
                dataframe['Activity'], dataframe['Session'], dataframe['Start Time'], dataframe['End Time'],
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
