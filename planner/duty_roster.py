from .utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_CLASS,
    DUTY_DURATION,
    DUTY_END_TIME,
    DUTY_IDEAL_CASE,
    DUTY_MIN_REQUIREMENT,
    DUTY_REQUIRED_FUNCTION,
    DUTY_RESTRICTED_FUNCTION,
    DUTY_SESSION,
    DUTY_START_TIME,
    DUTY_STAFF_PREFERENCE,
)

class DutyRoster:
    """
    A class responsible for managing the duty roster, including adding days and duties.
    """

    def __init__(self):
        self._duty_roster = {}

    @staticmethod
    def calculate_duration(start_time: str, end_time: str) -> float:
        """Calculate duty duration in hours from HHMM start and end times.

        Args:
            start_time (str): The start time in HHMM format, or a float/int representing HHMM.
            end_time (str): The end time in HHMM format, or a float/int representing HHMM.

        Returns:
            float: The duration of the duty in hours, accounting for overnight spans.
        """
        def normalize(time_value):
            if isinstance(time_value, float):
                if time_value.is_integer():
                    time_value = int(time_value)
                else:
                    raise ValueError(
                        f"Time values must be whole minutes in HHMM format, not {time_value}"
                    )
            if isinstance(time_value, int):
                time_value = str(time_value)
            if isinstance(time_value, str):
                if "." in time_value:
                    numeric = float(time_value)
                    if numeric.is_integer():
                        time_value = str(int(numeric))
                    else:
                        raise ValueError(
                            f"Time values must be whole minutes in HHMM format, not {time_value}"
                        )
                time_value = time_value.zfill(4)
                return time_value
            raise TypeError(
                f"Unsupported time type: {type(time_value).__name__}"
            )

        start_time = normalize(start_time)
        end_time = normalize(end_time)
        start_hour = int(start_time[:2])
        start_minute = int(start_time[2:])
        end_hour = int(end_time[:2])
        end_minute = int(end_time[2:])
        start_total_minutes = start_hour * 60 + start_minute
        end_total_minutes = end_hour * 60 + end_minute
        if end_total_minutes < start_total_minutes:
            end_total_minutes += 24 * 60
        duration_minutes = end_total_minutes - start_total_minutes
        return duration_minutes / 60

    def _add_day(self, day):
        if day not in self._duty_roster:
            self._duty_roster[day] = {}

    def add_duty(
        self,
        day,
        activity,
        class_name=None,
        session=None,
        start_time=None,
        end_time=None,
        min_requirement=0,
        ideal_case=0,
        required_function=None,
        restricted_function=None,
        staff_preference=None
    ):
        """Add a duty definition for a specific day.

        Args:
            day (str): The day key for the duty roster entry (e.g. "Monday_AM").
            activity (str): The duty name or activity label.
            class_name (str): The class name for the duty.
            session (str): The shift/session label (e.g. "AM" or "PM").
            start_time (str): Start time in HHMM format.
            end_time (str): End time in HHMM format.
            min_requirement (int): Minimum required staff count for the duty.
            ideal_case (int): Ideal total staff count for the duty.
            required_function (str, optional): Required staff function, if any.
            restricted_function (str, optional): Restricted staff function, if any.
            staff_preference (str, optional): Preference for teacher or temp assignment.
        """
        if day not in self._duty_roster:
            self._add_day(day)
        print(f'Adding duty: {activity} on {day} from {start_time} to {end_time}')
        self._duty_roster[day][activity] = {
            DUTY_CLASS: class_name if class_name is not None else "",
            DUTY_SESSION: session,
            DUTY_START_TIME: start_time,
            DUTY_END_TIME: end_time,
            DUTY_DURATION: self.calculate_duration(start_time=start_time, end_time=end_time) if start_time and end_time else 0,
            DUTY_MIN_REQUIREMENT: min_requirement,
            DUTY_IDEAL_CASE: ideal_case,
            DUTY_REQUIRED_FUNCTION: required_function,
            DUTY_RESTRICTED_FUNCTION: restricted_function,
            DUTY_STAFF_PREFERENCE: staff_preference,
            DUTY_ASSIGNEES: []
        }

    def get_duty_roster(self):
        return self._duty_roster
