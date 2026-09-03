import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .duty_roster import DutyRoster
from .person import Person
from .queue import Queue
from .staff_attributes import StaffAttributes
from .utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_ID,
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
    helper methods `_add_to_queue`, `_optimize_duty_assignment`, and `_assign_staff_to_duty`.
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

        all_staff_availability = self._get_staff_availability("AvailabilityList.xlsx")
        self._duty_roster = DutyRoster()
        self._get_duties_list_from_excel("DutiesBreakdown.xlsx")

        staff_queues = {}
        for staff_type, staff_list in all_staff_availability.items():
            queue = Queue()
            self._add_to_queue(queue, staff_list, staff_type=staff_type)
            staff_queues[staff_type] = queue

        schedule_state = self._optimize_duty_assignment(staff_queues)

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

    def _add_to_queue(self, queue, staff_list, staff_type: str = None):
        """Convert Excel rows into queue entries and normalize time values.

        Args:
            queue (Queue): The queue to populate with staff availability.
            staff_list (list): Rows containing day, session, name, start time, and end time.
            staff_type (str, optional): Default staff type for entries in this queue.
        """
        for row in staff_list:
            day = self._format_day_key(row[0], row[1])
            staff_name = str(row[2]).strip()
            start_time = str(Person.normalize_time(row[3])).zfill(4)
            end_time = str(Person.normalize_time(row[4])).zfill(4)
            row_staff_type = str(row[5]).strip() if len(row) > 5 and row[5] is not None and not (isinstance(row[5], float) and pd.isna(row[5])) else staff_type
            expected_capacity = self._parse_expected_capacity(row[6]) if len(row) > 6 else 1.0
            queue.add_to_queue(
                staff_member=staff_name,
                day=day,
                start_time=start_time,
                end_time=end_time,
                status=0,
                staff_type=row_staff_type or staff_type,
                expected_capacity=expected_capacity,
            )

    @staticmethod
    def _format_day_key(day, date) -> str:
        """Build a stable day key for both Excel dates and legacy sessions."""
        parsed_date = Scheduler._parse_schedule_date(date)
        if not pd.isna(parsed_date):
            date_key = parsed_date.strftime("%Y-%m-%d")
        else:
            date_key = str(date).replace(" ", "_")
        return f"{day}_{date_key}"

    @staticmethod
    def _parse_schedule_date(date):
        date_text = str(date).strip()
        is_iso_date = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}(?: .*)?", date_text))
        return pd.to_datetime(date, dayfirst=not is_iso_date, errors="coerce")

    @staticmethod
    def _validate_day_matches_date(day, date):
        parsed_date = Scheduler._parse_schedule_date(date)
        if pd.isna(parsed_date):
            return

        expected_day = parsed_date.day_name()
        if str(day).strip().casefold() != expected_day.casefold():
            raise ValueError(
                f"Duty day/date mismatch: '{day}' does not match "
                f"{parsed_date.strftime('%Y-%m-%d')} (expected {expected_day})."
            )

    @staticmethod
    def _order_queues_by_preference(staff_queues: Dict[str, Queue], preference: Optional[str]) -> List[Queue]:
        """Return staff queues ordered according to a preference string.

        Supports single or chained staff type preferences separated by ';' or ':' or ','.
        For example: 'teacher;temps;CH' or 'temps:CH:teacher'.
        Validates that all specified staff types exist in staff_queues.
        """
        if preference is None or (isinstance(preference, float) and pd.isna(preference)):
            return list(staff_queues.values())

        pref_str = str(preference).strip()
        if not pref_str or pref_str.lower() in ("no preference", "none", "nan"):
            return list(staff_queues.values())

        # Split by delimiters: semicolon, colon, or comma
        raw_tokens = re.split(r"[;:,]+", pref_str)
        tokens = [t.strip() for t in raw_tokens if t.strip()]

        if not tokens:
            return list(staff_queues.values())

        valid_keys = list(staff_queues.keys())
        ordered_keys = []

        for token in tokens:
            token_clean = token.lower()     # Ensures lowercase

            if token_clean in ("no preference", "none", "nan"):
                continue

            matched_key = None
            for k in valid_keys:
                k_clean = k.strip().lower()
                if k_clean == token_clean or k_clean.rstrip('s') == token_clean.rstrip('s'):
                    matched_key = k
                    break

            if matched_key is None:
                raise ValueError(
                    f"Invalid staff type '{token}' in staff preference '{preference}'. "
                    f"Valid staff types are: {valid_keys + ['No Preference']}"
                )

            if matched_key not in ordered_keys:
                ordered_keys.append(matched_key)

        # Append any remaining queues in their original order
        for k in valid_keys:
            if k not in ordered_keys:
                ordered_keys.append(k)

        return [staff_queues[k] for k in ordered_keys]

    @staticmethod
    def _order_duties_for_assignment(duties: dict, staff_queues: Dict[str, Queue], staff_attributes: StaffAttributes):
        """Return duties ordered by assignment priority.

        Duties requiring special functions or restrictions are assigned first so
        skilled staff are preserved for constrained assignments. Duties with an
        explicit staff preference are assigned before duties with no preference
        so preferred staff are not consumed by generic duties.

        This method considers the number of available staff across all queues for each duty,
        prioritizing duties with fewer qualified candidates.
        """
        queues_list = list(staff_queues.values())

        def count_available_for_duty(duty_info):
            """Count how many people in the queues can perform a given duty."""
            required_function = duty_info.get(DUTY_REQUIRED_FUNCTION)
            restricted_function = duty_info.get(DUTY_RESTRICTED_FUNCTION)

            def person_filter(person):
                if staff_attributes is None:
                    return True
                return (
                    staff_attributes.has_required_function(person.get_name(), required_function) and
                    not staff_attributes.has_restriction(person.get_name(), restricted_function)
                )

            return sum(1 for q in queues_list for p in q.get_list() if person_filter(p))

        def has_staff_preference(duty_info):
            preference = duty_info.get(DUTY_STAFF_PREFERENCE)
            if preference is None or (isinstance(preference, float) and pd.isna(preference)):
                return False
            return str(preference).strip().lower() not in ("", "no preference", "none", "nan")

        return sorted(
            duties.items(),
            key=lambda item: (
                0 if item[1].get(DUTY_REQUIRED_FUNCTION) or item[1].get(DUTY_RESTRICTED_FUNCTION) else 1,
                0 if has_staff_preference(item[1]) else 1,
                count_available_for_duty(item[1]),
                0 if item[1].get(DUTY_REQUIRED_FUNCTION) is None else -1,
                0 if item[1].get(DUTY_RESTRICTED_FUNCTION) is None else -1,
                -(item[1].get(DUTY_MIN_REQUIREMENT) or 0),
                -(item[1].get(DUTY_IDEAL_CASE) or 0),
                -(item[1].get(DUTY_DURATION) or 0),
            )
        )

    def _optimize_duty_assignment(
        self,
        staff_queues: Dict[str, Queue],
    ) -> ScheduleState:
        """
        Generate and evaluate candidate duty assignments to find the best distribution.

        This method performs a fixed number of iterations, each time shuffling the staff queues,
        assigning staff to every duty slot, and computing the selected fairness metric.
        The lowest-scoring valid assignment is retained and returned.

        Args:
            staff_queues (Dict[str, Queue]): Dictionary of queues by staff type.

        Returns:
            ScheduleState: Best schedule state found.
        """
        min_metric = float("inf")
        finalized_staff_queues = None
        final_roster = None
        last_error = None
        for i in range(100):
            duty_roster = copy.deepcopy(self._duty_roster.get_duty_roster())
            _staff_queues = {k: copy.deepcopy(q) for k, q in staff_queues.items()}
            for q in _staff_queues.values():
                q.shuffle()
            assignment_successful = True
            for day in duty_roster:
                day_start_queues = copy.deepcopy(_staff_queues)
                day_start_duties = copy.deepcopy(duty_roster[day])
                ordered_duties = self._order_duties_for_assignment(
                    duty_roster[day], _staff_queues, self._staff_attributes
                )

                assignment_orders = [ordered_duties]
                failed_index = None
                # Iterate through each assignment order and attempt to assign staff to duties
                for assignment_order in assignment_orders:
                    # Reset the staff queues and duty roster for the current day before attempting assignment
                    _staff_queues = copy.deepcopy(day_start_queues)
                    duty_roster[day] = copy.deepcopy(day_start_duties)
                    assignment_successful = True
                    # Attempt to assign staff to each duty in the current assignment order
                    for assignment_index, (duty_id, duty_info) in enumerate(assignment_order):
                        duty_info = duty_roster[day][duty_id]
                        assignment_result = self._assign_staff_to_duty(
                            day, duty_info, _staff_queues,
                            duty_info[DUTY_MIN_REQUIREMENT], ideal_case=False,
                            duties_for_day=duty_roster[day],
                            duty_name=duty_info.get(DUTY_ACTIVITY, "Duty"))
                        if assignment_result is not True:
                            # If assignment fails, mark the assignment as unsuccessful and record the index of the failed duty
                            assignment_successful = False
                            failed_index = assignment_index
                            last_error = assignment_result
                            break
                    if assignment_successful:
                        break

                    if assignment_order is not ordered_duties:
                        # If the failed assignment order is not the original ordered duties, 
                        # skip generating new orders to avoid infinite loops.
                        continue

                    failed_duty = assignment_order[failed_index][1]
                    preference = failed_duty.get(DUTY_STAFF_PREFERENCE)
                    is_generic_duty = not (
                        failed_duty.get(DUTY_REQUIRED_FUNCTION)
                        or failed_duty.get(DUTY_RESTRICTED_FUNCTION)
                    ) and (
                        preference is None
                        or str(preference).strip().lower() in ("", "no preference", "none", "nan")
                    )
                    if is_generic_duty:
                        for target_index in range(failed_index):
                            # Generate a new assignment order by moving the failed duty to a different position in the list
                            retry_order = list(ordered_duties)
                            retry_order.insert(target_index, retry_order.pop(failed_index))
                            assignment_orders.append(retry_order)

                if not assignment_successful:
                    # Stop assigning duties for the current day and move to the next iteration.
                    break

                for duty_id, duty_info in self._order_duties_for_assignment(duty_roster[day], _staff_queues, self._staff_attributes):
                    if duty_info[DUTY_MIN_REQUIREMENT] < duty_info[DUTY_IDEAL_CASE]:
                        self._assign_staff_to_duty(
                            day, duty_info, _staff_queues,
                            duty_info[DUTY_IDEAL_CASE] - duty_info[DUTY_MIN_REQUIREMENT],
                            ideal_case=True,
                            duties_for_day=duty_roster[day],
                            duty_name=duty_info.get(DUTY_ACTIVITY, "Duty"))

            if not assignment_successful:
                # Move to next iteration if no staff assigned to a duty
                continue

            lunch_check_result = self._lunch_provider_satisfied(_staff_queues, duty_roster)
            if lunch_check_result is not True:
                last_error = lunch_check_result
                continue

            combined_people = [p for q in _staff_queues.values() for p in q.get_list()]
            days = sorted(duty_roster.keys())

            if self._fairness_mode == "week":
                metric = sum(q.find_std_deviation() for q in _staff_queues.values())

            else:
                import math

                daily_stds = []
                for day_key in days:
                    values = [
                        p.get_hours_worked_by_day().get(day_key, 0) / p.get_expected_capacity(day_key)
                        for p in combined_people
                    ]
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
                finalized_staff_queues = copy.deepcopy(_staff_queues)
                final_roster = duty_roster

        if final_roster is None:
            if last_error:
                raise ValueError(last_error)
            raise ValueError(
                "Unable to generate a valid schedule after multiple attempts. All generated schedules failed."
            )

        return ScheduleState(final_roster, finalized_staff_queues)

    def _lunch_provider_satisfied(self, staff_queues: Dict[str, Queue], duty_roster: dict):
        """Validate that staff receive the configured minimum rest slots in the lunch window.

        Returns:
            True if satisfied, otherwise an error string with failure details.
        """
        for q in staff_queues.values():
            for person in q.get_list():
                for day_key in sorted(duty_roster.keys()):
                    if not self._is_lunch_window_applicable(person, day_key):
                        continue
                    rest_slots = self._count_rest_slots_during_window(
                        person,
                        day_key,
                        self._lunch_break_start,
                        self._lunch_break_end,
                    )
                    if rest_slots < self._lunch_break_min_rest_slots:
                        return (
                            f"Lunch break validation failed for {person.get_name()} on {day_key}. "
                            f"Required {self._lunch_break_min_rest_slots} rest slots between "
                            f"{self._lunch_break_start} and {self._lunch_break_end}, but found only {rest_slots}."
                        )
        return True

    def _is_lunch_window_applicable(self, person, day_key):
        """Return True when the person is in school for the entire lunch window.

        Adhoc staff who are only present for part of the lunch period are excluded
        so they are not required to keep the configured minimum lunch rest.
        """
        availability = person.get_availability(day_key)
        if not availability:
            return False

        lunch_start = getattr(self, "_lunch_break_start", None)
        lunch_end = getattr(self, "_lunch_break_end", None)
        if not lunch_start or not lunch_end:
            return False

        start_index = Person.time_to_index(
            lunch_start, person._base_start_minutes, person._slot_minutes
        )
        end_index = Person.time_to_index(
            lunch_end, person._base_start_minutes, person._slot_minutes
        )
        if start_index < 0 or end_index > len(availability) or start_index >= end_index:
            return False

        lunch_slots = availability[start_index:end_index]
        return all(status in (0, 1) for status in lunch_slots)

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

    def _would_violate_lunch_rest(self, person: Person, day: str, duty_start_time: str, duty_end_time: str) -> bool:
        """
        Check whether assigning this duty to the person would cause them to have
        fewer than the required minimum rest slots during the configured lunch window.

        The check only applies when the person is in school for the entire lunch
        window. Adhoc staff who are present for only part of lunch are excluded.

        Args:
            person (Person): The staff candidate.
            day (str): Day identifier.
            duty_start_time (str): Start time of duty in HHMM format.
            duty_end_time (str): End time of duty in HHMM format.

        Returns:
            bool: True if assignment would reduce remaining lunch rest slots below
                the minimum required, False otherwise.
        """
        min_rest_slots = getattr(self, "_lunch_break_min_rest_slots", 0)
        if min_rest_slots <= 0:
            return False

        if not hasattr(self, "_lunch_break_start") or not hasattr(self, "_lunch_break_end"):
            return False

        if not self._is_lunch_window_applicable(person, day):
            return False

        avail = person.get_availability(day)
        if not avail:
            return False

        start_idx = Person.time_to_index(duty_start_time, person._base_start_minutes, person._slot_minutes)
        end_idx = Person.time_to_index(duty_end_time, person._base_start_minutes, person._slot_minutes)
        lunch_start_idx = Person.time_to_index(self._lunch_break_start, person._base_start_minutes, person._slot_minutes)
        lunch_end_idx = Person.time_to_index(self._lunch_break_end, person._base_start_minutes, person._slot_minutes)

        overlap_start = max(start_idx, lunch_start_idx)
        overlap_end = min(end_idx, lunch_end_idx)

        # If duty overlaps with the lunch window, simulate occupying those slots
        if overlap_start < overlap_end:
            current_lunch_slots = avail[lunch_start_idx:lunch_end_idx]
            simulated_slots = list(current_lunch_slots)
            for i in range(overlap_start - lunch_start_idx, overlap_end - lunch_start_idx):
                if 0 <= i < len(simulated_slots):
                    simulated_slots[i] = 1

            remaining_rest = simulated_slots.count(0)
            if remaining_rest < min_rest_slots:
                return True

        return False

    def _assign_staff_to_duty(
        self,
        day: str,
        duty_info: dict,
        staff_queues: Dict[str, Queue],
        required_count: int,
        ideal_case: bool = False,
        duties_for_day: Optional[dict] = None,
        duty_name: Optional[str] = None,
    ) -> Union[bool, str]:
        """
        Assign available staff members to a specified duty slot.

        Selects staff members from the staff queues based on staff preferences,
        required/restricted qualifications, and lunch-break rest constraints. Assigned staff
        have their workloads and schedules updated.
        """
        duty_name = duty_name or duty_info.get(DUTY_ACTIVITY, "Duty")
        preference = duty_info.get(DUTY_STAFF_PREFERENCE)
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

        ordered_queues = self._order_queues_by_preference(staff_queues, preference)

        for _ in range(int(required_count)):
            # This function will try to select a person based on the preference order.
            def select_person(person_filter):
                for q in ordered_queues:
                    person = q.select_available_person(day, duty_info[DUTY_START_TIME], duty_info[DUTY_END_TIME], person_filter)
                    if person is not None:
                        return person
                return None

            # --- Selection Logic with Proactive Lunch Rest Protection ---
            selected_person = None
            duty_start_time = duty_info[DUTY_START_TIME]
            duty_end_time = duty_info[DUTY_END_TIME]
            duty_start_minutes = Person._time_to_minutes(duty_start_time)
            lunch_start_minutes = Person._time_to_minutes(self._lunch_break_start)
            lunch_end_minutes = Person._time_to_minutes(self._lunch_break_end)

            is_during_lunch = lunch_start_minutes <= duty_start_minutes < lunch_end_minutes

            # Pass 1: Ideal candidate — qualified, maintains minimum lunch rest budget, and not working immediately before
            def pass1_filter(person):
                if not base_person_filter(person):
                    return False
                if self._would_violate_lunch_rest(person, day, duty_start_time, duty_end_time):
                    return False
                if is_during_lunch and self._was_working_before(person, day, duty_start_time):
                    return False
                return True

            selected_person = select_person(pass1_filter)

            # Pass 2: Qualified candidate who maintains minimum lunch rest budget (relaxing the 'working before' check)
            if not selected_person:
                def pass2_filter(person):
                    return (
                        base_person_filter(person)
                        and not self._would_violate_lunch_rest(person, day, duty_start_time, duty_end_time)
                    )

                selected_person = select_person(pass2_filter)

            # Pass 3: Fallback if strictly necessary (relaxing lunch rest budget so highly constrained slots still attempt assignment)
            if not selected_person:
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
                return self._format_insufficient_staff_error(
                    day=day,
                    duty_info=duty_info,
                    staff_queues=staff_queues,
                    duties_for_day=duties_for_day,
                    duty_name=duty_name
                )

            if ideal_case:
                return True
        return True

    def _format_insufficient_staff_error(
        self,
        day: str,
        duty_info: dict,
        staff_queues: Dict[str, Queue],
        duties_for_day: Optional[dict] = None,
        duty_name: Optional[str] = None,
    ) -> str:
        """
        Format a detailed diagnostic error message when staff assignment fails.

        Provides a breakdown of the target duty requirements, concurrent duties in the
        same timeframe with their assigned staff, and the status of all staff
        (engaged, disqualified, or unavailable).
        """
        duty_name = duty_name or duty_info.get(DUTY_ACTIVITY, "Duty")
        start_time = duty_info.get(DUTY_START_TIME, "??")
        end_time = duty_info.get(DUTY_END_TIME, "??")
        req_count = duty_info.get(DUTY_MIN_REQUIREMENT, 1)
        ideal_count = duty_info.get(DUTY_IDEAL_CASE, req_count)
        req_func = duty_info.get(DUTY_REQUIRED_FUNCTION)
        restr_func = duty_info.get(DUTY_RESTRICTED_FUNCTION)
        pref = duty_info.get(DUTY_STAFF_PREFERENCE, "No Preference")

        currently_assigned = [p.get_name() for p in duty_info.get(DUTY_ASSIGNEES, [])]
        assigned_str = ", ".join(currently_assigned) if currently_assigned else "None"
        still_needed = max(0, req_count - len(currently_assigned))

        lines = [
            f"Unable to find sufficient staff for {duty_name} on {day} from {start_time} to {end_time}.",
            "",
            "--- Target Duty Details ---",
            f"  Activity: {duty_name}",
            f"  Time: {start_time} - {end_time}",
            f"  Requirements: Min: {req_count} | Ideal: {ideal_count} | Still Needed: {still_needed}",
            f"  Preferences / Functions: Preference: {pref} | Required Function: {req_func} | Restriction: {restr_func}",
            f"  Currently Assigned: {assigned_str}",
        ]

        try:
            target_start_min = Person._time_to_minutes(start_time)
            target_end_min = Person._time_to_minutes(end_time)
        except Exception:
            target_start_min = 0
            target_end_min = 0

        # 1. Concurrent Duties in the same timeframe
        if duties_for_day:
            concurrent_duties = []
            for d_id, d_info in duties_for_day.items():
                try:
                    d_start = Person._time_to_minutes(d_info[DUTY_START_TIME])
                    d_end = Person._time_to_minutes(d_info[DUTY_END_TIME])
                    if max(target_start_min, d_start) < min(target_end_min, d_end):
                        concurrent_duties.append(d_info)
                except Exception:
                    continue

            # Sort concurrent duties by start time and id
            concurrent_duties.sort(key=lambda d: (d.get(DUTY_START_TIME, ""), d.get(DUTY_ID, 0)))

            lines.append("")
            lines.append(f"--- Concurrent Duties in Timeframe ({start_time} - {end_time}) ---")
            total_min_required = sum(d.get(DUTY_MIN_REQUIREMENT, 0) for d in concurrent_duties)
            lines.append(f"  Total Concurrent Duties: {len(concurrent_duties)} | Total Minimum Staff Required: {total_min_required}")
            for idx, d in enumerate(concurrent_duties, 1):
                d_act = d.get(DUTY_ACTIVITY, "Duty")
                d_id = d.get(DUTY_ID, "")
                d_id_str = f" (ID: {d_id})" if d_id else ""
                d_s = d.get(DUTY_START_TIME, "")
                d_e = d.get(DUTY_END_TIME, "")
                d_min = d.get(DUTY_MIN_REQUIREMENT, 0)
                d_ideal = d.get(DUTY_IDEAL_CASE, 0)
                d_rf = d.get(DUTY_REQUIRED_FUNCTION)
                d_rf_str = f" | Req: {d_rf}" if d_rf else ""
                d_restr = d.get(DUTY_RESTRICTED_FUNCTION)
                d_restr_str = f" | Restr: {d_restr}" if d_restr else ""
                d_assignees = [p.get_name() for p in d.get(DUTY_ASSIGNEES, [])]
                d_assignees_str = ", ".join(d_assignees) if d_assignees else "None"
                lines.append(f"  {idx}. {d_act}{d_id_str} [{d_s} - {d_e}] | Min: {d_min}, Ideal: {d_ideal}{d_rf_str}{d_restr_str} | Assigned: {d_assignees_str}")

        # 2. Staff Status during the timeframe
        all_staff = [(role, p) for role, q in staff_queues.items() for p in q.get_list()]

        lunch_start_minutes = Person._time_to_minutes(self._lunch_break_start) if hasattr(self, "_lunch_break_start") else 0
        lunch_end_minutes = Person._time_to_minutes(self._lunch_break_end) if hasattr(self, "_lunch_break_end") else 0
        is_during_lunch = lunch_start_minutes <= target_start_min < lunch_end_minutes

        engaged_staff = []
        free_disqualified = []
        free_eligible = []
        unavailable_staff = []

        for role, person in all_staff:
            name = person.get_name()
            if person.check_availability(day, start_time, end_time):
                # Free in time slot, check why not chosen
                reasons = []
                if req_func and hasattr(self, "_staff_attributes") and not self._staff_attributes.has_required_function(name, req_func):
                    reasons.append(f"missing required function '{req_func}'")
                if restr_func and hasattr(self, "_staff_attributes") and self._staff_attributes.has_restriction(name, restr_func):
                    reasons.append(f"has restriction '{restr_func}'")
                if hasattr(self, "_would_violate_lunch_rest") and self._would_violate_lunch_rest(person, day, start_time, end_time):
                    reasons.append("assignment would exhaust required minimum lunch rest slots")
                elif is_during_lunch and hasattr(self, "_was_working_before") and self._was_working_before(person, day, start_time):
                    reasons.append("worked immediately prior in lunch window")

                if reasons:
                    free_disqualified.append(f"  - {role}: {name} (Free, but disqualified: {', '.join(reasons)})")
                else:
                    free_eligible.append(f"  - {role}: {name} (Free & Eligible)")
            else:
                avail = person.get_availability(day)
                start_idx = Person.time_to_index(start_time, person._base_start_minutes, person._slot_minutes)
                end_idx = Person.time_to_index(end_time, person._base_start_minutes, person._slot_minutes)

                slots = avail[start_idx:end_idx] if 0 <= start_idx < len(avail) else []
                if 1 in slots:
                    # Find what they are assigned to
                    assigned_to = []
                    if duties_for_day:
                        for d_id, d_info in duties_for_day.items():
                            if any(a.get_name() == name for a in d_info.get(DUTY_ASSIGNEES, [])):
                                try:
                                    d_s = Person._time_to_minutes(d_info[DUTY_START_TIME])
                                    d_e = Person._time_to_minutes(d_info[DUTY_END_TIME])
                                    if max(target_start_min, d_s) < min(target_end_min, d_e):
                                        assigned_to.append(f"{d_info.get(DUTY_ACTIVITY)} [{d_info.get(DUTY_START_TIME)}-{d_info.get(DUTY_END_TIME)}]")
                                except Exception:
                                    continue
                    assigned_msg = f"Assigned to: {', '.join(assigned_to)}" if assigned_to else "Working on another duty"
                    engaged_staff.append(f"  - {role}: {name} ({assigned_msg})")
                elif all(s == -1 for s in slots) or not slots:
                    unavailable_staff.append(f"  - {role}: {name} (Not in school / Not scheduled)")
                else:
                    unavailable_staff.append(f"  - {role}: {name} (Partially unavailable during slot)")

        lines.append("")
        lines.append(f"--- Staff Status during Timeframe ({start_time} - {end_time}) ---")
        lines.append(f"  Available & Engaged Staff ({len(engaged_staff)}):")
        lines.extend(engaged_staff or ["    (None)"])
        if free_eligible:
            lines.append(f"  Available & Eligible Staff ({len(free_eligible)}):")
            lines.extend(free_eligible)
        if free_disqualified:
            lines.append(f"  Available but Disqualified Staff ({len(free_disqualified)}):")
            lines.extend(free_disqualified)
        lines.append(f"  Unavailable / Absent Staff ({len(unavailable_staff)}):")
        lines.extend(unavailable_staff or ["    (None)"])

        return "\n".join(lines)

    @staticmethod
    def _get_staff_availability(file_name: str) -> Dict[str, List]:
        """
        Load staff availability from an Excel file with any number of sheets.

        Rows from all sheets are grouped by their "Staff Type". If the same
        staff type appears in multiple sheets, the rows are combined.

        Args:
            file_name (str): Path to the Excel file containing staff availability.

        Returns:
            Dict[str, List]: Mapping of staff type to list of rows.
        """
        staff_dict: Dict[str, List] = {}

        excel_file = pd.ExcelFile(file_name)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            if "Staff Type" not in df.columns:
                raise ValueError(
                    f"Sheet '{sheet_name}' does not contain a 'Staff Type' column."
                )

            if "Expected Capacity" not in df.columns:
                df["Expected Capacity"] = 1.0

            for staff_type, group in df.groupby("Staff Type"):
                rows = group.values.tolist()
                staff_dict.setdefault(staff_type, []).extend(rows)

        return staff_dict

    @staticmethod
    def _parse_expected_capacity(value) -> float:
        """Parse an expected capacity cell, defaulting blank/NaN values to 1.0."""
        if value is None:
            return 1.0
        if isinstance(value, float) and pd.isna(value):
            return 1.0
        if isinstance(value, str) and not value.strip():
            return 1.0
        return Person._validate_expected_capacity(value)

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
            self._validate_day_matches_date(day, date)
            day_key = self._format_day_key(day, date)

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