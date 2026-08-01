import copy
from typing import List, Tuple
from enum import Enum

import pandas as pd

from .duty_roster import DutyRoster
from .queue import Queue
from .staff_attributes import StaffAttributes


class StaffPreference(Enum):
    TEACHER_FIRST = "Teacher First"
    TEMP_FIRST = "Temp First"
    NO_PREFERENCE = "No Preference"


class Scheduler:
    """
    Assigns staff to duty slots based on availability and writes the final roster to Excel.

    The scheduler performs the following steps when instantiated:
      1. Loads staff availability from `AvailabilityList.xlsx`.
      2. Creates a `DutyRoster` and populates it with days from teacher availability.
      3. Loads duty definitions from `DutiesBreakdown.xlsx`.
      4. Builds separate queues for teachers and temps and marks each person's availability.
      5. Runs a multi-iteration optimization loop that shuffles the queues, assigns staff to duties,
         and selects the best schedule based on the lowest combined workload standard deviation.
      6. Writes the selected roster and work distribution to `teacher_schedule_with_duties.xlsx`.

    This class is intentionally designed as an orchestration layer; most detailed logic lives in the
    helper methods `_add_to_queue`, `_optimize_duty_assignment`, `_assign_staff_to_duty`, and
    `_write_roster_to_excel`.
    """

    def __init__(self):
        """
        Initialize the Scheduler and execute the duty planning workflow.

        The constructor performs the full scheduling process on instantiation:
          - load staff availability from Excel
          - initialize the duty roster for each day
          - load duty definitions from Excel
          - build teacher and temp queues based on availability
          - optimize assignment over multiple candidate schedules
          - write the selected roster and distribution to Excel
        """
        self._staff_attributes = StaffAttributes()
        self._get_staff_attributes_from_excel(
            "StaffAttributes.xlsx"
        )

        teachers_list, temps_list = self._get_staff_availability("AvailabilityList.xlsx")
        self._duty_roster = DutyRoster()
        self._get_duties_list_from_excel("DutiesBreakdown.xlsx")
        teacher_list = Queue()
        temp_list = Queue()
        self._add_to_queue(teacher_list, teachers_list)
        self._add_to_queue(temp_list, temps_list)

        best_schedule, finalized_teacher_list, finalized_temp_list = self._optimize_duty_assignment(
            teacher_list,
            temp_list
        )
        self._write_roster_to_excel(best_schedule, finalized_teacher_list, finalized_temp_list)

    def _get_staff_attributes_from_excel(self, file_name):
        """Load staff required and restricted functions from Excel into StaffAttributes.

        Args:
            file_name (str): Path to the StaffAttributes Excel workbook.
        """
        df_required = pd.read_excel(
            file_name,
            sheet_name="Special Functions"
        )
        for staff_name, function in zip(
            df_required["Staff Name"],
            df_required["Special Function"]):

            self._staff_attributes.add_required_function(
                staff_name.strip(),
                function.strip()
            )
        df_restricted = pd.read_excel(
            file_name,
            sheet_name="Restrictions"
        )
        for staff_name, restriction in zip(
                df_restricted["Staff Name"],
                df_restricted["Restrictions"]):

            self._staff_attributes.add_restricted_function(
                staff_name.strip(),
                restriction.strip()
            )

    def _add_to_queue(self, queue, staff_list):
        for row in staff_list:
            day = f"{row[0]}_{str(row[1]).replace(' ', '_')}"
            staff_name = row[2].strip()
            start_time = str(row[3]).zfill(4)
            end_time = str(row[4]).zfill(4)
            queue.add_to_queue(
                staff_member=staff_name,
                day=day,
                start_time=start_time,
                end_time=end_time,
                status=0
            )

    def _optimize_duty_assignment(self, teacher_list, temp_list):
        """
        Generate and evaluate candidate duty assignments to find the best distribution.

        This method performs a fixed number of iterations, each time shuffling the teacher and temp queues,
        assigning staff to every duty slot, and computing the combined standard deviation of workload ratios.
        The lowest-scoring assignment is retained and returned.

        Args:
            teacher_list (Queue): Queue of available teachers.
            temp_list (Queue): Queue of available temps.

        Returns:
            tuple: (best_roster, best_teacher_list, best_temp_list)
        """
        min_std_deviation = float("inf")
        finalized_teacher_list = None
        finalized_temp_list = None
        final_roster = None
        for _ in range(100):
            _teacher_list = copy.deepcopy(teacher_list)
            _temp_list = copy.deepcopy(temp_list)
            duty_roster = copy.deepcopy(self._duty_roster.get_duty_roster())
            for day in duty_roster:
                _teacher_list.shuffle()
                _temp_list.shuffle()
                for duty_name, duty_info in duty_roster[day].items():
                    self._assign_staff_to_duty(
                        day, duty_info, _teacher_list, _temp_list, duty_info["min_requirement"], ideal_case=False)
                for duty_name, duty_info in duty_roster[day].items():
                    if duty_info["min_requirement"] < duty_info["ideal_case"]:
                        self._assign_staff_to_duty(
                            day, duty_info, _teacher_list, _temp_list,
                            duty_info["ideal_case"] - duty_info["min_requirement"],
                            ideal_case=True)
            sum_of_std_deviation = _teacher_list.find_std_deviation() + _temp_list.find_std_deviation()
            if sum_of_std_deviation < min_std_deviation:
                min_std_deviation = sum_of_std_deviation
                finalized_teacher_list = copy.deepcopy(_teacher_list)
                finalized_temp_list = copy.deepcopy(_temp_list)
                final_roster = duty_roster
        return final_roster, finalized_teacher_list, finalized_temp_list

    def _assign_staff_to_duty(self, day, duty_info, teacher_list, temp_list, required_count, ideal_case: bool):
        """
        Assigns staff to a single duty until the required headcount is reached.

        Args:
            day (str): Normalized day identifier for the duty (e.g. "Monday_AM").
            duty_info (dict): Duty metadata including start_time, end_time, assignees, min_requirement, and ideal_case.
            teacher_list (Queue): Queue of available teachers.
            temp_list (Queue): Queue of available temps.
            required_count (int): Number of staff to assign for this pass.
            ideal_case (bool): Whether this assignment is filling optional ideal positions.

        Notes:
            - Teachers are chosen first, then temps if no teacher is available.
            - If `ideal_case` is True, a single assignment is attempted and the method returns early.
            - If `ideal_case` is False and no staff is available, a ValueError is raised.
        """
        preference = duty_info.get(
            "staff_preference",
            StaffPreference.NO_PREFERENCE.value
        )
        if preference is None or (isinstance(preference, float) and pd.isna(preference)):
            preference = StaffPreference.NO_PREFERENCE.value

        required_function = duty_info.get("required_function")
        restricted_function = duty_info.get("restricted_function")

        # Define a filter function to check if a person meets the required and restricted function criteria.
        person_filter = lambda person: (
            self._staff_attributes.has_required_function(
                person.get_name(),
                required_function
            )
            and
            not self._staff_attributes.has_restriction(
                person.get_name(),
                restricted_function
            )
        )

        for _ in range(required_count):
            selected_teacher = None
            selected_temp = None
            if preference == StaffPreference.TEACHER_FIRST.value:
                selected_teacher = teacher_list.select_available_person(
                    day,
                    duty_info["start_time"],
                    duty_info["end_time"],
                    person_filter=person_filter
                )
                if selected_teacher:
                    duty_info["assignees"].append(selected_teacher)
                else:
                    # If no teacher is available, try to assign a temp
                    selected_temp = temp_list.select_available_person(
                        day,
                        duty_info["start_time"],
                        duty_info["end_time"],
                        person_filter=person_filter
                    )
                    if selected_temp:
                        duty_info["assignees"].append(selected_temp)
            elif preference == StaffPreference.TEMP_FIRST.value:
                selected_temp = temp_list.select_available_person(
                    day,
                    duty_info["start_time"],
                    duty_info["end_time"],
                    person_filter=person_filter
                )
                if selected_temp:
                    duty_info["assignees"].append(selected_temp)
                else:
                    # If no temp is available, try to assign a teacher
                    selected_teacher = teacher_list.select_available_person(
                        day,
                        duty_info["start_time"],
                        duty_info["end_time"],
                        person_filter=person_filter
                    )
                    if selected_teacher:
                        duty_info["assignees"].append(selected_teacher)
            elif preference == StaffPreference.NO_PREFERENCE.value:
                # If no preference is specified, select the staff member with the lowest workload ratio
                selected_teacher = teacher_list.select_available_person(
                    day,
                    duty_info["start_time"],
                    duty_info["end_time"],
                    person_filter=person_filter
                )
                selected_temp = temp_list.select_available_person(
                    day,
                    duty_info["start_time"],
                    duty_info["end_time"],
                    person_filter=person_filter
                )
                if selected_teacher and selected_temp:
                    if (
                        selected_teacher.get_work_capacity_ratio()
                        <= selected_temp.get_work_capacity_ratio()
                    ):
                        duty_info["assignees"].append(selected_teacher)
                    else:
                        duty_info["assignees"].append(selected_temp)
                elif selected_teacher:
                    duty_info["assignees"].append(selected_teacher)
                elif selected_temp:
                    duty_info["assignees"].append(selected_temp)
            else:
                raise ValueError(f"Unknown staff preference: {preference}")

            if ideal_case:
                return

            if not selected_teacher and not selected_temp:
                raise ValueError(
                    f"Unable to find sufficient staff for "
                    f"{duty_info['start_time']} to {duty_info['end_time']} on {day}"
                )

    @staticmethod
    def _write_roster_to_excel(roster: dict, finalized_teacher_list: Queue, finalized_temp_list: Queue) -> None:
        """
        Write the finalized duty roster and work distribution to an Excel file.

        Args:
            roster (dict): The selected duty schedule keyed by day.
            finalized_teacher_list (Queue): Final teacher queue with updated workload state.
            finalized_temp_list (Queue): Final temp queue with updated workload state.

        Output:
            Creates `teacher_schedule_with_duties.xlsx` with two sheets:
              - Duty Roster
              - Work Distribution
        """
        teachers_by_day = {}
        for day in roster:
            teachers_by_day[day] = []
            for duty in roster[day]:
                assignees = roster[day][duty]["assignees"]
                teachers_for_duty = [assignee.get_name() for assignee in assignees] + ["NA"] * (6 - len(assignees))
                teachers_by_day[day].append((duty, teachers_for_duty))
        data_for_excel = []
        for day, duties in teachers_by_day.items():
            for duty, duty_teachers in duties:
                data_for_excel.append([day, duty] + duty_teachers)
        people = []
        number_of_duties_taken = []
        hours_worked_list = []
        hours_in_school_list = []
        for person in finalized_temp_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_work_capacity_ratio())
            hours_worked_list.append(person.get_hours_worked())
            hours_in_school_list.append(person.get_hours_in_school())
        for person in finalized_teacher_list.get_list():
            people.append(person.get_name())
            number_of_duties_taken.append(person.get_work_capacity_ratio())
            hours_worked_list.append(person.get_hours_worked())
            hours_in_school_list.append(person.get_hours_in_school())
        work_distribution = pd.DataFrame(
            {
                "Person": people,
                "Work To Capacity": number_of_duties_taken,
                "Hours Worked": hours_worked_list,
                "Hours In School": hours_in_school_list,
            }
        )
        df_roster = pd.DataFrame(data_for_excel,
                                 columns=["Day", "Duty", "Teacher 1", "Teacher 2", "Teacher 3", "Teacher 4",
                                          "Teacher 5", "Teacher 6"])
        with pd.ExcelWriter("teacher_schedule_with_duties.xlsx", engine="xlsxwriter") as writer:
            df_roster.to_excel(writer, sheet_name="Duty Roster", index=False)
            work_distribution.to_excel(writer, sheet_name="Work Distribution", index=False)
        print("Data has been written to teacher_schedule_with_duties.xlsx")

    @staticmethod
    def _get_staff_availability(file_name: str) -> Tuple[List, List]:
        """
        Load staff availability from an Excel file with two sheets: "Teachers" and "Temps".

        Args:
            file_name (str): Path to the Excel file containing staff availability.

        Returns:
            tuple:  (teachers_list, temps_list) where each list contains rows of
                    [Day, Session, Name, Start Time, End Time].
        """
        df_teachers = pd.read_excel(file_name, sheet_name="Teachers")
        df_temps = pd.read_excel(file_name, sheet_name="Temps")

        return (
            df_teachers.values.tolist(),
            df_temps.values.tolist()
        )

    def _get_duties_list_from_excel(self, file_name):
        """
        Load duty definitions from an Excel file and add them to the roster.

        Args:
            file_name (str): Path to the Excel file containing duty definitions.

        Notes:
            Each duty row must include Activity, Session, Start Time, End Time,
            Minimum Requirement, and Ideal Case.
        """
        dataframe = pd.read_excel(file_name)
        for (
            day,
            date,
            activity,
            session,
            start_time,
            end_time,
            min_requirement,
            ideal_case,
            required_function,
            restricted_function,
            staff_preference
        ) in zip(
            dataframe["Day"],
            dataframe["Date"],
            dataframe["Activity"],
            dataframe["Session"],
            dataframe["Start Time"],
            dataframe["End Time"],
            dataframe["Minimum Requirement"],
            dataframe["Ideal Case"],
            dataframe["Required Function"],
            dataframe["Restricted Function"],
            dataframe["Staff Preference"]
        ):
            day_key = f"{day}_{str(date).replace(' ', '_')}"

            self._duty_roster.add_duty(
                day=day_key,
                activity=activity,
                session=session,
                start_time=start_time,
                end_time=end_time,
                min_requirement=min_requirement,
                ideal_case=ideal_case,
                required_function=None if pd.isna(required_function) else required_function,
                restricted_function=None if pd.isna(restricted_function) else restricted_function,
                staff_preference=staff_preference
            )
