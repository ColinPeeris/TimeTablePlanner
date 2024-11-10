import copy
import math
import pandas as pd
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
        self._availability = []
        self._days_assigned = []

    def get_name(self) -> str:
        """
        Returns the name of the person.

        Returns:
            str: The name of the person.
        """
        return self._name

    def get_availability(self) -> List[str]:
        """
        Returns the list of days the person is available to take on duties.

        Returns:
            list: A list of days the person is available for duty.
        """
        return self._availability

    def set_availability(self, day: str) -> None:
        """
        Adds a day to the person's availability list.

        Args:
            day (str): The day the person becomes available for duty.
        """
        self._availability.append(day)

    def check_availability(self, day: str) -> bool:
        """
        Checks if the person is available on a specific day.

        Args:
            day (str): The day to check availability for.

        Returns:
            bool: True if the person is available on the specified day, False otherwise.
        """
        return True if day in self._availability else False

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

    def add_to_queue(self, person: Person) -> None:
        """
        Adds a `Person` object to the queue, updating their availability if they are already in the queue.

        Args:
            person (Person): The `Person` object to add to the queue.
        """
        # Iterate through the queue to find if the person already exists in the queue
        for index, entry in enumerate(self._queue):
            # Check if the current entry in the queue has the same name as the given person
            if entry.get_name() == person.get_name():
                # If the person already exists, update their availability in the queue
                for day_available in person.get_availability():
                    # Add each available day of the person to the existing entry in the queue
                    self._queue[index].set_availability(day=day_available)
                # Once the availability is updated, exit the method
                return
        # If the person was not found in the queue, add them to the queue
        self._queue.append(person)

    def select_available_person(self, day: str) -> Optional[Person]:
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
        capacity = len(person.get_availability())
        work_load = person.get_number_of_duties_taken()
        assert capacity >= work_load
        return float(work_load) / float(capacity)

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
        teachers, temps = self._get_data_from_excel('AvailabilityList.xlsx')
        slots_to_fill = teachers.keys()

        teacher_list = Queue()
        temp_list = Queue()
        for slot in slots_to_fill:
            # make sure the slot exists in the teachers queue and the temp queue.
            assert slot in teachers
            assert slot in temps
            for teacher in teachers[slot]:
                person_to_add = Person(teacher)
                person_to_add.set_availability(slot)
                teacher_list.add_to_queue(person=person_to_add)
            for temp in temps[slot]:
                person_to_add = Person(temp)
                person_to_add.set_availability(slot)
                temp_list.add_to_queue(person=person_to_add)

        min_std_deviation = 99999
        success_count = 0
        finalized_teacher_list = None
        finalized_temp_list = None
        final_duty_list = None

        for i in range(0, 100):
            # make copies of teacher and temp queues
            _teacher_list = copy.deepcopy(teacher_list)
            _temp_list = copy.deepcopy(temp_list)

            # randomly shuffle teacher and temp lists
            _teacher_list.shuffle()
            _temp_list.shuffle()

            # create a duty list
            duty_list = {}
            try:
                for slot in slots_to_fill:
                    assert slot in teachers
                    assert slot in temps
                    duty_list[slot] = self._create_duties_per_slot(available_teachers=_teacher_list,
                                                                   available_temps=_temp_list,
                                                                   slot_to_fill=slot)
            except AssertionError as e:
                print(f'Assertion failed during duty creation for slot: {slot}. Error: {str(e)}')
                continue  # Continue to the next slot if an assertion fails
            except Exception as e:
                print(f'An error occurred while creating duties for slot: {slot}. Error: {str(e)}')
                continue  # Continue to the next slot if a general error occurs

            sum_of_std_deviation = _teacher_list.find_std_deviation() + _temp_list.find_std_deviation()
            if sum_of_std_deviation < min_std_deviation:
                min_std_deviation = sum_of_std_deviation
                finalized_teacher_list = copy.deepcopy(_teacher_list)
                finalized_temp_list = copy.deepcopy(_temp_list)
                assert duty_list is not None
                final_duty_list = duty_list

            success_count += 1

        print(f'Number of successful runs: {success_count}')

        if (finalized_temp_list is not None) and (finalized_teacher_list is not None):
            print('------------------------------------')
            print('Print Best Results:')
            for slot in slots_to_fill:
                print(slot)
                for person in final_duty_list[slot]:
                    print(person.get_name())
                print('\n')
            for person in finalized_temp_list.get_list():
                print(f'Person: {person.get_name()}, Number Of Duties: {person.get_number_of_duties_taken()}')
            for person in finalized_teacher_list.get_list():
                print(f'Person: {person.get_name()}, Number Of Duties: {person.get_number_of_duties_taken()}')
            self._write_results_to_excel(teacher_list=finalized_teacher_list,
                                         temp_list=finalized_temp_list,
                                         duty_list=final_duty_list,
                                         slots_to_fill=slots_to_fill)

    @staticmethod
    def _write_results_to_excel(teacher_list: Queue, temp_list: Queue, duty_list: dict, slots_to_fill: List) -> None:
        """
        Writes the finalized duty assignments and work distribution to an Excel file.

        Args:
            teacher_list (Queue): The final list of teachers assigned to duties.
            temp_list (Queue): The final list of temporary staff (temps) assigned to duties.
            duty_list (dict): A dictionary mapping time slots to assigned duties.
            slots_to_fill (list): A list of time slots that need to be filled with duties.

        Saves two sheets in the 'output.xlsx' Excel file:
            - 'duty_list': Contains the names of teachers assigned to each slot.
            - 'duties_distribution': Lists the number of duties taken by each person.
        """
        slots = []
        teacher_1 = []
        teacher_2 = []
        teacher_3 = []
        teacher_4 = []
        teacher_5 = []
        teacher_6 = []
        for slot in slots_to_fill:
            slots.append(slot)
            assert len(duty_list[slot]) > 4
            teacher_1.append(duty_list[slot][0].get_name())
            teacher_2.append(duty_list[slot][1].get_name())
            teacher_3.append(duty_list[slot][2].get_name())
            teacher_4.append(duty_list[slot][3].get_name())
            teacher_5.append(duty_list[slot][4].get_name())
            if len(duty_list[slot]) > 5:
                teacher_6.append(duty_list[slot][5].get_name())
            else:
                teacher_6.append('NA')
        duty_list_data = pd.DataFrame(
            {'Time Slot': slots,
             'Teacher 1': teacher_1,
             'Teacher 2': teacher_2,
             'Teacher 3': teacher_3,
             'Teacher 4': teacher_4,
             'Teacher 5': teacher_5,
             'Teacher 6': teacher_6,
             }
        )

        people = []
        number_of_duties_taken = []
        for person in temp_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_number_of_duties_taken())
        for person in teacher_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_number_of_duties_taken())
        work_distribution = pd.DataFrame(
            {'Person': people,
             'Number of Duties Taken': number_of_duties_taken,
             }
        )

        writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
        frames = {'duty_list': duty_list_data,
                  'duties_distribution': work_distribution}
        for sheet in frames.keys():
            frame = frames[sheet]
            frame.to_excel(writer, sheet_name=sheet)
        writer.save()

    @staticmethod
    def _get_by_am_pm(df, am: bool) -> Dict:
        """
        Extracts availability data from a DataFrame for either AM or PM slots.

        Args:
            df (pandas.DataFrame): The DataFrame containing the availability data.
            am (bool): If True, extracts data for AM slots. If False, extracts data for PM slots.

        Returns:
            dict: A dictionary with the day tags (e.g., 'Monday_AM', 'Tuesday_PM') as keys and
                  lists of names of available people as values.
        """
        output = {}
        data = df.to_numpy()
        rows, cols = data.shape
        for row in range(0, rows):
            if am:
                # create the am tag
                day_tag = data[row][0] + '_' + str(data[row][1]).split(' ')[0] + '_AM'
            else:
                # create the pm tag
                day_tag = data[row][0] + '_' + str(data[row][1]).split(' ')[0] + '_PM'
            names = []
            for col in range(2, cols):
                if str(data[row][col]) != 'nan':
                    name = data[row][col]
                    name = name.replace(" ", "")
                    names.append(name)
            if len(names) > 0:
                output[day_tag] = names
        return output

    def _get_data_by_sheet(self, df_am, df_pm) -> Dict:
        """
        Combines the AM and PM availability data for teachers and temps from their respective DataFrames.

        Args:
            df_am (pandas.DataFrame): DataFrame containing AM availability for teachers or temps.
            df_pm (pandas.DataFrame): DataFrame containing PM availability for teachers or temps.

        Returns:
            dict: A dictionary combining the availability data for both AM and PM time slots.
        """
        return {**self._get_by_am_pm(df_am, am=True), **self._get_by_am_pm(df_pm, am=False)}

    def _get_data_from_excel(self, file_name) -> Tuple[Dict, Dict]:
        """
        Reads availability data for teachers and temps from the given Excel file.

        Args:
            file_name (str): The name of the Excel file to read data from.

        Returns:
            tuple: A tuple containing two dictionaries:
                   - The first dictionary contains the availability data for teachers.
                   - The second dictionary contains the availability data for temps.
        """
        df_teachers_am = pd.read_excel(file_name, sheet_name='Teachers_AM')
        df_teachers_pm = pd.read_excel(file_name, sheet_name='Teachers_PM')
        df_temps_am = pd.read_excel(file_name, sheet_name='Temps_AM')
        df_temps_pm = pd.read_excel(file_name, sheet_name='Temps_PM')
        return self._get_data_by_sheet(df_am=df_teachers_am, df_pm=df_teachers_pm), \
            self._get_data_by_sheet(df_am=df_temps_am, df_pm=df_temps_pm)

    @staticmethod
    def _create_duties_per_slot(available_teachers: Queue, available_temps: Queue, slot_to_fill: str) -> List:
        """
        Creates a list of assigned duties for a specific time slot (AM or PM).

        The method assigns teachers and temps to different roles (PreN, N, K for AM and N, K for PM) based
        on their availability. It ensures that there are sufficient available staff for each role, and attempts
        to balance the workload.

        Args:
            available_teachers (Queue): A queue of teachers available for duty.
            available_temps (Queue): A queue of temporary staff (temps) available for duty.
            slot_to_fill (str): The time slot (e.g., 'Monday_AM', 'Tuesday_PM') for which duties are being created.

        Returns:
            list: A list of Person objects representing the assigned teachers and temps for the slot.

        Raises:
            AssertionError: If no suitable teacher or temp is available for a required role.
        """
        """
        What is needed:
        Morning:
        - PreN: 2
        - N: 2
        - K: 2
        Afternoon:
        - N: 2-3
        - K: 2
        """
        duty_list = []

        N_Person1 = available_teachers.select_available_person(day=slot_to_fill)
        assert N_Person1 is not None, f'No teacher available for N in {slot_to_fill}'
        N_Person2 = available_temps.select_available_person(day=slot_to_fill)
        if N_Person2 is None:
            N_Person2 = available_teachers.select_available_person(day=slot_to_fill)
            assert N_Person2 is not None, f'Nobody teacher/temp available for N in {slot_to_fill}'
        duty_list.append(N_Person1)
        duty_list.append(N_Person2)

        K_Person1 = available_teachers.select_available_person(day=slot_to_fill)
        assert K_Person1 is not None, f'No teacher available for K in {slot_to_fill}'
        K_Person2 = available_temps.select_available_person(day=slot_to_fill)
        if K_Person2 is None:
            K_Person2 = available_teachers.select_available_person(day=slot_to_fill)
            assert K_Person2 is not None, f'Nobody teacher/temp available for K in {slot_to_fill}'
        duty_list.append(K_Person1)
        duty_list.append(K_Person2)

        if 'AM' in slot_to_fill:
            PreN_Person1 = available_teachers.select_available_person(day=slot_to_fill)
            assert PreN_Person1 is not None, f'No teacher available for PreN in {slot_to_fill}'
            PreN_Person2 = available_temps.select_available_person(day=slot_to_fill)
            if PreN_Person2 is None:
                PreN_Person2 = available_teachers.select_available_person(day=slot_to_fill)
                assert PreN_Person2 is not None, f'Nobody teacher/temp available for PreN in {slot_to_fill}'
            duty_list.append(PreN_Person1)
            duty_list.append(PreN_Person2)
        elif 'PM' in slot_to_fill:
            N_Person3 = available_temps.select_available_person(day=slot_to_fill)
            if N_Person3 is None:
                N_Person3 = available_teachers.select_available_person(day=slot_to_fill)
            if N_Person3 is None:
                print(f'Warning: No 3rd teacher / temp available for {slot_to_fill}')
            else:
                duty_list.append(N_Person3)
        else:
            assert True, f'Invalid entry {slot_to_fill}'

        return duty_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scheduler = Scheduler()

# To run it, run the following:
# python .\main.py
