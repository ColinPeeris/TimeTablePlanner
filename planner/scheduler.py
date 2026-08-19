import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd

from .duty_roster import DutyRoster
from .person import Person
from .queue import Queue
from .staff_attributes import StaffAttributes
from .utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_ACTIVITY,
    DUTY_DURATION,
    DUTY_END_TIME,
    DUTY_IDEAL_CASE,
    DUTY_MIN_REQUIREMENT,
    DUTY_REQUIRED_FUNCTION,
    DUTY_RESTRICTED_FUNCTION,
    DUTY_START_TIME,
    DUTY_STAFF_PREFERENCE,
)
from .utils.configs import (
    FAIRNESS_MODE,
    VALID_FAIRNESS_MODES,
    LUNCH_BREAK_START,
    LUNCH_BREAK_END,
    LUNCH_BREAK_MIN_REST_SLOTS,
)
from .schedule_state import ScheduleState
from .state_serializer import StateSerializer
from report_generator import ReportGenerator


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

    def __init__(self, fairness_mode: str = None, days_to_schedule: List[str] = None, previous_roster: dict = None):
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
        if days_to_schedule is None and previous_roster is not None:
            raise ValueError("If previous_roster is provided, days_to_schedule must also be provided.")
        if days_to_schedule is not None and previous_roster is None:
            raise ValueError("If days_to_schedule is provided, previous_roster must also be provided.")

        self._staff_attributes = StaffAttributes()
        self._get_staff_attributes_from_excel(
            "StaffAttributes.xlsx"
        )

        # Fairness mode controls how fairness is evaluated during optimization.
        # Supported values:
        #  - 'week'    : original behaviour, minimize weekly stddev (teachers+temps)
        #  - 'day_sum' : minimize sum of daily stddevs across the week
        #  - 'day_max' : minimize the worst-day stddev (minimax)
        if fairness_mode is None:
            fairness_mode = FAIRNESS_MODE
        if fairness_mode not in VALID_FAIRNESS_MODES:
            raise ValueError(f"Unknown fairness mode: {fairness_mode}")
        self._fairness_mode = fairness_mode

        self._lunch_break_start = LUNCH_BREAK_START
        self._lunch_break_end = LUNCH_BREAK_END
        self._lunch_break_min_rest_slots = LUNCH_BREAK_MIN_REST_SLOTS

        teachers_list, temps_list = self._get_staff_availability("AvailabilityList.xlsx")
        self._duty_roster = DutyRoster()
        self._get_duties_list_from_excel("DutiesBreakdown.xlsx")
        teacher_list = Queue()
        temp_list = Queue()
        self._add_to_queue(teacher_list, teachers_list)
        self._add_to_queue(temp_list, temps_list)

        schedule_state = self._optimize_duty_assignment(
            teacher_list,
            temp_list
        )

        if schedule_state is None:
            raise ValueError(
                "Unable to find a valid roster that satisfies the configured lunch-break "
                "requirements. Please verify the lunch-break window and minimum rest slots."
            )

        # Persist state companion file
        excel_filename = "teacher_schedule_with_duties.xlsx"
        state_filename = excel_filename.replace('.xlsx', '.state')
        StateSerializer.save(schedule_state, state_filename)

        # Generate human-readable workbook (presentation only)
        ReportGenerator(schedule_state, filename=excel_filename).generate()

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
        """Convert Excel rows into queue entries and normalize time values.

        Args:
            queue (Queue): The queue to populate with staff availability.
            staff_list (list): Rows containing day, session, name, start time, and end time.
        """
        for row in staff_list:
            day = f"{row[0]}_{str(row[1]).replace(' ', '_')}"
            staff_name = row[2].strip()
            start_time = str(Person.normalize_time(row[3])).zfill(4)
            end_time = str(Person.normalize_time(row[4])).zfill(4)
            queue.add_to_queue(
                staff_member=staff_name,
                day=day,
                start_time=start_time,
                end_time=end_time,
                status=0
            )

    @staticmethod
    def _order_duties_for_assignment(duties, teacher_list, temp_list, staff_attributes):
        """Return duties ordered by assignment priority.

        Duties requiring special functions or restrictions are assigned first so
        skilled staff are preserved for constrained assignments.

        This method now considers the number of available staff for each duty,
        prioritizing duties with fewer qualified candidates.
        """
        def count_available_for_duty(duty_info):
            """Count how many people in the queues can perform a given duty."""
            required_function = duty_info.get(DUTY_REQUIRED_FUNCTION)
            restricted_function = duty_info.get(DUTY_RESTRICTED_FUNCTION)

            def person_filter(person):
                return (
                    staff_attributes.has_required_function(person.get_name(), required_function) and
                    not staff_attributes.has_restriction(person.get_name(), restricted_function)
                )

            teacher_count = sum(1 for p in teacher_list.get_list() if person_filter(p))
            temp_count = sum(1 for p in temp_list.get_list() if person_filter(p))
            return teacher_count + temp_count

        return sorted(
            duties.items(),
            key=lambda item: (
                # New: Prioritize duties with fewer available staff.
                count_available_for_duty(item[1]),
                # Existing criteria follow
                0 if item[1].get(DUTY_REQUIRED_FUNCTION) is None else -1,
                0 if item[1].get(DUTY_RESTRICTED_FUNCTION) is None else -1,
                -(item[1].get(DUTY_MIN_REQUIREMENT) or 0),
                -(item[1].get(DUTY_IDEAL_CASE) or 0),
                -(item[1].get(DUTY_DURATION) or 0),
            )
        )

    def _optimize_duty_assignment(self, teacher_list, temp_list):
        """
        Generate and evaluate candidate duty assignments to find the best distribution.

        This method performs a fixed number of iterations, each time shuffling the teacher and temp queues,
        assigning staff to every duty slot, and computing the selected fairness metric.
        The lowest-scoring valid assignment is retained and returned.

        Args:
            teacher_list (Queue): Queue of available teachers.
            temp_list (Queue): Queue of available temps.

        Returns:
            tuple: (best_roster, best_teacher_list, best_temp_list)
        """
        min_metric = float("inf")
        finalized_teacher_list = None
        finalized_temp_list = None
        final_roster = None
        last_error = None
        for i in range(100):
            duty_roster = copy.deepcopy(self._duty_roster.get_duty_roster())
            _teacher_list = copy.deepcopy(teacher_list)
            _temp_list = copy.deepcopy(temp_list)
            _teacher_list.shuffle()
            _temp_list.shuffle()
            assignment_successful = True
            for day in duty_roster:
                for duty_id, duty_info in self._order_duties_for_assignment(duty_roster[day], _teacher_list, _temp_list, self._staff_attributes):
                    assignment_result = self._assign_staff_to_duty(
                        day, duty_info, _teacher_list, _temp_list,
                        duty_info[DUTY_MIN_REQUIREMENT], ideal_case=False,
                        duties_for_day=duty_roster[day],
                        duty_name=duty_info.get(DUTY_ACTIVITY, "Duty"))
                    if assignment_result is not True:
                        assignment_successful = False
                        last_error = assignment_result
                        break

                if not assignment_successful:
                    # Stop assigning duties for the current day and move to the next iteration.
                    break

                for duty_id, duty_info in self._order_duties_for_assignment(duty_roster[day], _teacher_list, _temp_list, self._staff_attributes):
                    # If the duty has fewer than the ideal number of assignees, attempt to assign additional staff to
                    # reach the ideal case.
                    if duty_info[DUTY_MIN_REQUIREMENT] < duty_info[DUTY_IDEAL_CASE]:
                        # For ideal case, we don't care about the return value as it's optional.
                        self._assign_staff_to_duty(
                            day, duty_info, _teacher_list, _temp_list,
                            duty_info[DUTY_IDEAL_CASE] - duty_info[DUTY_MIN_REQUIREMENT],
                            ideal_case=True,
                            duties_for_day=duty_roster[day],
                            duty_name=duty_info.get(DUTY_ACTIVITY, "Duty"))

            if not assignment_successful:
                # Skip to the next iteration if the assignment was not successful.
                continue

            # Enforce any configured post-assignment checks before evaluating fairness.
            lunch_check_result = self._lunch_provider_satisfied(_teacher_list, duty_roster)
            if lunch_check_result is not True:
                # If the check fails, it returns an error string. Store it.
                last_error = lunch_check_result
                continue

            # Choose fairness metric according to configuration
            combined_people = _teacher_list.get_list() + _temp_list.get_list()
            days = sorted(duty_roster.keys())

            if self._fairness_mode == "week":
                # Original behaviour: combined std deviation for teachers + temps
                metric = _teacher_list.find_std_deviation() + _temp_list.find_std_deviation()

            else:
                # Compute daily std deviations across combined_people for each day
                import math

                daily_stds = []
                for day_key in days:
                    values = [p.get_hours_worked_by_day().get(day_key, 0) for p in combined_people]
                    if not values:
                        daily_stds.append(0.0)
                        continue
                    mean = sum(values) / len(values)
                    var = sum((v - mean) ** 2 for v in values) / len(values)
                    daily_stds.append(math.sqrt(var))

                if self._fairness_mode == "day_sum":
                    metric = sum(daily_stds)
                elif self._fairness_mode == "day_max":
                    metric = max(daily_stds) if daily_stds else 0.0
                else:
                    raise ValueError(f"Unknown fairness_mode: {self._fairness_mode}")

            if metric < min_metric:
                min_metric = metric
                finalized_teacher_list = copy.deepcopy(_teacher_list)
                finalized_temp_list = copy.deepcopy(_temp_list)
                final_roster = duty_roster

        if final_roster is None:
            if last_error:
                raise ValueError(last_error)
            raise ValueError(
                "Unable to generate a valid schedule after multiple attempts. All generated schedules failed."
            )

        return ScheduleState(final_roster, finalized_teacher_list, finalized_temp_list)

    def _lunch_provider_satisfied(self, teacher_list, duty_roster):
        """Validate that teachers receive the configured minimum rest slots in the lunch window.

        Returns:
            True if satisfied, otherwise an error string with failure details.
        """
        for teacher in teacher_list.get_list():
            for day_key in sorted(duty_roster.keys()):
                if not self._is_lunch_window_applicable(teacher, day_key):
                    continue
                rest_slots = self._count_rest_slots_during_window(
                    teacher,
                    day_key,
                    self._lunch_break_start,
                    self._lunch_break_end,
                )
                if rest_slots < self._lunch_break_min_rest_slots:
                    return (
                        f"Lunch break validation failed for {teacher.get_name()} on {day_key}. "
                        f"Required {self._lunch_break_min_rest_slots} rest slots between "
                        f"{self._lunch_break_start} and {self._lunch_break_end}, but found only {rest_slots}."
                    )
        return True

    @staticmethod
    def _is_lunch_window_applicable(person, day_key):
        availability = person.get_availability(day_key)
        if not availability:
            return False
        return any(status in (0, 1) for status in availability)

    @staticmethod
    def _count_rest_slots_during_window(person, day_key, start_time, end_time):
        availability = person.get_availability(day_key)
        if not availability:
            return 0
        start_index = Person.time_to_index(start_time, person._base_start_minutes, person._slot_minutes)
        end_index = Person.time_to_index(end_time, person._base_start_minutes, person._slot_minutes)
        start_index = max(start_index, 0)
        end_index = min(end_index, len(availability))
        if start_index >= end_index:
            return 0
        return sum(1 for status in availability[start_index:end_index] if status == 0)

    @staticmethod
    def _was_working_before(person: Person, day: str, duty_start_time: str) -> bool:
        """Check if the person was working in the slot just before the duty."""
        start_index = person.time_to_index(duty_start_time)
        if start_index == 0:
            return False
        availability = person.get_availability(day)
        if not availability or start_index >= len(availability):
            return False
        # Status of 1 means "working"
        return availability[start_index - 1] == 1

    def _assign_staff_to_duty(
        self,
        day: str,
        duty_info: dict,
        teacher_list: Queue,
        temp_list: Queue,
        required_count: int,
        ideal_case: bool,
        duties_for_day: Optional[dict] = None,
        duty_name: Optional[str] = None,
    ) -> Union[bool, str]:
        """
        Assign available staff members to a specified duty slot.

        Selects staff members from the teacher and temp queues based on staff preferences,
        required/restricted qualifications, and lunch-break rest constraints. Assigned staff
        have their workloads and schedules updated.

        Args:
            day (str): The day identifier for the duty (e.g., 'Wednesday_2026-08-19_00:00:00').
            duty_info (dict): Dictionary containing duty metadata (time, assignees, requirements).
            teacher_list (Queue): Queue of teacher staff members.
            temp_list (Queue): Queue of temporary staff members.
            required_count (int): Number of staff members to assign.
            ideal_case (bool): Whether this assignment is for optional ideal capacity (True)
                or mandatory minimum requirement (False).
            duties_for_day (Optional[dict]): All duties scheduled for the given day. Defaults to None.
            duty_name (Optional[str]): Human-readable name of the duty activity. Defaults to None.

        Returns:
            Union[bool, str]: True if the requested number of staff was successfully assigned
                (or if ideal_case is True), otherwise an error string detailing the failure.
        """
        duty_name = duty_name or duty_info.get(DUTY_ACTIVITY, "Duty")

        preference = duty_info.get(
            DUTY_STAFF_PREFERENCE,
            StaffPreference.NO_PREFERENCE.value
        )
        if preference is None or (isinstance(preference, float) and pd.isna(preference)):
            preference = StaffPreference.NO_PREFERENCE.value

        required_function = duty_info.get(DUTY_REQUIRED_FUNCTION)
        restricted_function = duty_info.get(DUTY_RESTRICTED_FUNCTION)

        # Base filter for skills and restrictions
        base_person_filter = lambda person: (
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

        for _ in range(int(required_count)):
            # This function will try to select a person based on the preference order.
            def select_person(person_filter):
                if preference == StaffPreference.TEACHER_FIRST.value:
                    person = teacher_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    if not person:
                        person = temp_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    return person
                elif preference == StaffPreference.TEMP_FIRST.value:
                    person = temp_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    if not person:
                        person = teacher_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    return person
                elif preference == StaffPreference.NO_PREFERENCE.value:
                    # Default to Teacher First for no preference
                    person = teacher_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    if not person:
                        person = temp_list.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    return person
                else:
                    raise ValueError(f"Unknown staff preference: {preference}")

            # --- Selection Logic ---
            selected_person = None
            duty_start_time = duty_info[DUTY_START_TIME]
            duty_start_minutes = Person._time_to_minutes(duty_start_time)
            lunch_start_minutes = Person._time_to_minutes(self._lunch_break_start)
            lunch_end_minutes = Person._time_to_minutes(self._lunch_break_end)

            is_during_lunch = lunch_start_minutes <= duty_start_minutes < lunch_end_minutes

            if is_during_lunch:
                # --- Two-Pass Selection for lunch window ---
                # 1. First Pass: Try to find a rested person
                def rested_filter(person):
                    return base_person_filter(person) and not self._was_working_before(person, day, duty_start_time)

                selected_person = select_person(rested_filter)

                # 2. Second Pass: If no rested person, find anyone available
                if not selected_person:
                    selected_person = select_person(base_person_filter)
            else:
                # --- Single-Pass Selection outside lunch window ---
                selected_person = select_person(base_person_filter)

            # --- Assignment ---
            if selected_person:
                duty_info[DUTY_ASSIGNEES].append(selected_person)
                selected_person.add_duty(
                    day, duty_name, duty_info
                )
            else:
                # No one could be found, even with the fallback.
                if ideal_case:
                    return True # It's okay to not fill ideal slots
                return f"Unable to find sufficient staff for {duty_name} on {day} from {duty_start_time} to {duty_info[DUTY_END_TIME]}"

            if ideal_case:
                return True
        return True

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
                assignees = roster[day][duty][DUTY_ASSIGNEES]
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
    def _write_roster_to_excel_2(
        roster: dict,
        finalized_teacher_list: Queue,
        finalized_temp_list: Queue,
    ) -> None:

        import pandas as pd

        ###############################################################
        # Build Duty Roster sheet
        ###############################################################

        roster_rows = []

        for day in sorted(roster.keys()):

            duties = sorted(
                roster[day].items(),
                key=lambda x: x[1][DUTY_START_TIME]
            )

            for duty_id, duty_info in duties:

                teachers = [
                    p.get_name()
                    for p in duty_info[DUTY_ASSIGNEES]
                ]

                teachers += [""] * (6 - len(teachers))

                start = duty_info[DUTY_START_TIME]
                end = duty_info[DUTY_END_TIME]

                duration = (
                    Person.time_to_index(end)
                    - Person.time_to_index(start)
                ) * 0.5

                roster_rows.append([
                    day,
                    start,
                    end,
                    duration,
                    duty_info[DUTY_ACTIVITY],
                    *teachers
                ])

        df_roster = pd.DataFrame(
            roster_rows,
            columns=[
                "Day",
                "Start",
                "End",
                "Hours",
                "Duty",
                "Teacher 1",
                "Teacher 2",
                "Teacher 3",
                "Teacher 4",
                "Teacher 5",
                "Teacher 6",
            ],
        )

        ###############################################################
        # Build Work Distribution sheet
        ###############################################################

        people_rows = []

        people = (
            finalized_teacher_list.get_list()
            + finalized_temp_list.get_list()
        )

        all_days = sorted(roster.keys())

        for person in people:

            worked = person.get_hours_worked_by_day()
            rests = person.get_rest_periods_by_day()

            row = {
                "Person": person.get_name(),
                "Capacity": round(person.get_work_capacity_ratio(), 2),
                "Hours Worked": person.get_hours_worked(),
                "Hours In School": person.get_hours_in_school(),
            }

            total_rest = 0

            for day in all_days:

                work = worked.get(day, 0)
                rest = rests.get(day, 0)

                row[f"{day} Work"] = work
                row[f"{day} Rest"] = rest

                total_rest += rest

            row["Total Rest"] = total_rest

            people_rows.append(row)

        df_summary = pd.DataFrame(people_rows)

        ###############################################################
        # Write workbook
        ###############################################################

        with pd.ExcelWriter(
            "teacher_schedule_with_duties.xlsx",
            engine="xlsxwriter",
        ) as writer:

            df_roster.to_excel(
                writer,
                sheet_name="Duty Roster",
                index=False,
            )

            df_summary.to_excel(
                writer,
                sheet_name="Work Distribution",
                index=False,
            )

            workbook = writer.book

            header_format = workbook.add_format({
                "bold": True,
                "bg_color": "#D9EAD3",
                "border": 1,
                "align": "center",
            })

            centre_format = workbook.add_format({
                "align": "center",
            })

            for sheet_name, dataframe in {
                "Duty Roster": df_roster,
                "Work Distribution": df_summary,
            }.items():

                worksheet = writer.sheets[sheet_name]

                worksheet.freeze_panes(1, 0)

                for col_num, value in enumerate(dataframe.columns):
                    worksheet.write(0, col_num, value, header_format)

                    width = max(
                        len(str(value)),
                        dataframe.iloc[:, col_num].astype(str).map(len).max()
                    ) + 2

                    worksheet.set_column(
                        col_num,
                        col_num,
                        width,
                        centre_format,
                    )

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
        class_col = dataframe["Class"] if "Class" in dataframe.columns else [""] * len(dataframe)
        for (
            day,
            date,
            activity,
            class_name,
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
            class_col,
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

            normalized_start_time = str(Person.normalize_time(start_time)).zfill(4)
            normalized_end_time = str(Person.normalize_time(end_time)).zfill(4)
            self._duty_roster.add_duty(
                day=day_key,
                activity=activity,
                class_name=class_name,
                session=session,
                start_time=normalized_start_time,
                end_time=normalized_end_time,
                min_requirement=min_requirement,
                ideal_case=ideal_case,
                required_function=None if pd.isna(required_function) else required_function,
                restricted_function=None if pd.isna(restricted_function) else restricted_function,
                staff_preference=staff_preference
            )
