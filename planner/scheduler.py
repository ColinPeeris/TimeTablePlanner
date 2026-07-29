import copy
from typing import List, Tuple

import pandas as pd

from .duty_roster import DutyRoster
from .queue import Queue


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
    helper methods `_add_to_queue_for_slot`, `_optimize_duty_assignment`, `_assign_staff_to_duty`, and
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
        teachers_am_list, teachers_pm_list, temps_am_list, temps_pm_list = self._get_staff_availability(
            "AvailabilityList.xlsx"
        )
        self._duty_roster = DutyRoster()
        for slot in teachers_am_list:
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"
            self._duty_roster.add_day(day)
        self._get_duties_list_from_excel("DutiesBreakdown.xlsx")
        teacher_list = Queue()
        temp_list = Queue()
        # Add staff members to the queue for each time slot
        self._add_to_queue_for_slot(teacher_list, teachers_am_list, "0900", "1400")
        self._add_to_queue_for_slot(teacher_list, teachers_pm_list, "1400", "1800")
        self._add_to_queue_for_slot(temp_list, temps_am_list, "0900", "1400")
        self._add_to_queue_for_slot(temp_list, temps_pm_list, "1400", "1800")
        best_schedule, finalized_teacher_list, finalized_temp_list = self._optimize_duty_assignment(
            teacher_list,
            temp_list
        )
        self._write_roster_to_excel(best_schedule, finalized_teacher_list, finalized_temp_list)

    def _add_to_queue_for_slot(self, queue, slot_list, start_time, end_time):
        """
        Add staff availability to a queue for a given time slot.

        Args:
            queue (Queue): The queue to populate with available staff.
            slot_list (list): Rows of availability data, where each row contains day, session, and staff names.
            start_time (str): The slot start time in HHMM.
            end_time (str): The slot end time in HHMM.

        Notes:
            The day string is normalized by replacing spaces in the session label with underscores.
            Empty or NaN staff entries are ignored.
        """
        for slot in slot_list:
            day = f"{slot[0]}_{str(slot[1]).replace(' ', '_')}"
            staff_members = [staff_member.strip() for staff_member in slot[2:] if pd.notna(staff_member)]
            for staff_member in staff_members:
                queue.add_to_queue(staff_member=staff_member, day=day, start_time=start_time, end_time=end_time,
                                   status=0)

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
        for _ in range(required_count):
            selected_teacher = teacher_list.select_available_person(
                day, duty_info["start_time"], duty_info["end_time"]
            )
            selected_temp = None
            if selected_teacher:
                duty_info["assignees"].append(selected_teacher)
            else:
                selected_temp = temp_list.select_available_person(
                    day, duty_info["start_time"], duty_info["end_time"]
                )
                if selected_temp:
                    duty_info["assignees"].append(selected_temp)
            if ideal_case:
                return
            if not selected_teacher and not selected_temp:
                raise ValueError(
                    f"Unable to find sufficient staff for {duty_info['start_time']} to {duty_info['end_time']} on {day}"
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
    def _get_staff_availability(file_name) -> Tuple[List, List, List, List]:
        """
        Read staff availability sheets from an Excel file.

        Args:
            file_name (str): Path to the Excel file containing availability data.

        Returns:
            tuple: Four lists representing Teachers_AM, Teachers_PM, Temps_AM, Temps_PM.
        """
        df_teachers_am = pd.read_excel(file_name, sheet_name="Teachers_AM")
        df_teachers_pm = pd.read_excel(file_name, sheet_name="Teachers_PM")
        df_temps_am = pd.read_excel(file_name, sheet_name="Temps_AM")
        df_temps_pm = pd.read_excel(file_name, sheet_name="Temps_PM")
        return df_teachers_am.values.tolist(), df_teachers_pm.values.tolist(), df_temps_am.values.tolist(), df_temps_pm.values.tolist()

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
        for activity, session, start_time, end_time, min_requirement, ideal_case in zip(
                dataframe["Activity"], dataframe["Session"], dataframe["Start Time"], dataframe["End Time"],
                dataframe["Minimum Requirement"], dataframe["Ideal Case"]):
            self._duty_roster.add_duty(
                activity=activity,
                session=session,
                start_time=start_time,
                end_time=end_time,
                min_requirement=min_requirement,
                ideal_case=ideal_case
            )
