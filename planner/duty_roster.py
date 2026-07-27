class DutyRoster:
    """
    A class responsible for managing the duty roster, including adding days and duties.
    """

    def __init__(self):
        self._duty_roster = {}

    @staticmethod
    def calculate_duration(start_time: str, end_time: str) -> float:
        if isinstance(start_time, int):
            start_time = str(start_time)
        if isinstance(end_time, int):
            end_time = str(end_time)
        start_time = start_time.zfill(4)
        end_time = end_time.zfill(4)
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

    def add_day(self, day):
        if day not in self._duty_roster:
            self._duty_roster[day] = {}

    def add_duty(self, activity, session, start_time, end_time, min_requirement, ideal_case):
        for day in self._duty_roster:
            self._duty_roster[day][activity] = {
                "session": session,
                "start_time": start_time,
                "end_time": end_time,
                "duration": self.calculate_duration(start_time=start_time, end_time=end_time),
                "min_requirement": min_requirement,
                "ideal_case": ideal_case,
                "assignees": []
            }

    def get_duty_roster(self):
        return self._duty_roster
