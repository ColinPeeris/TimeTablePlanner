import re

import pandas as pd

from planner.person import Person
from planner.utils.constants import (
    DUTY_ASSIGNEES,
    DUTY_CLASS,
    DUTY_END_TIME,
    DUTY_START_TIME,
)


class ReportGenerator:

    def __init__(
        self,
        roster,
        teachers,
        temps,
        filename="teacher_schedule_with_duties.xlsx",
    ):
        self._roster = roster
        self._teachers = teachers
        self._temps = temps
        self._filename = filename

        self._people = (
            teachers.get_list() +
            temps.get_list()
        )

    def generate(self):
        duty_roster = self.create_duty_roster()
        work_distribution = self.create_work_distribution()
        self._teacher_timetables = self.create_teacher_timetable()
        self._timetable_sheets = {}

        with pd.ExcelWriter(
            self._filename,
            engine="xlsxwriter",
        ) as writer:

            duty_roster.to_excel(
                writer,
                sheet_name="Duty Roster",
                index=False,
            )

            activity_breakdown = self.create_activity_breakdown_by_class()
            activity_breakdown.to_excel(
                writer,
                sheet_name="Activity Breakdown",
                index=False,
            )

            work_distribution.to_excel(
                writer,
                sheet_name="Work Distribution",
                index=False,
            )

            for day, dataframe in self._teacher_timetables.items():
                sheet_name = self._sanitize_sheet_name(day)
                sheet_name = self._ensure_unique_sheet_name(sheet_name)
                self._timetable_sheets[sheet_name] = dataframe

                dataframe.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

            self._format_workbook(writer)

    def _ensure_unique_sheet_name(self, sheet_name: str) -> str:
        """Return a unique sheet name by appending a numeric suffix if needed."""
        if sheet_name not in self._timetable_sheets:
            return sheet_name

        suffix = 1
        base = sheet_name[:31]
        while True:
            suffix_name = f"{base[:31 - len(str(suffix)) - 1]}_{suffix}"
            if suffix_name not in self._timetable_sheets:
                return suffix_name
            suffix += 1

    @staticmethod
    def _sanitize_sheet_name(sheet_name: str) -> str:
        """Make a string safe for Excel worksheet names."""
        sanitized = re.sub(r"[\[\]\\:\*\?/]+", "_", sheet_name)
        return sanitized[:31] if len(sanitized) > 31 else sanitized

    def create_duty_roster(self) -> pd.DataFrame:
        """
        Create the Duty Roster dataframe.

        Returns
        -------
        pd.DataFrame
            Columns:
                Day
                Start
                End
                Duration (Hours)
                Duty
                Teacher 1 ... Teacher 6
        """

        rows = []

        for day in sorted(self._roster.keys()):

            # Sort duties chronologically
            duties = sorted(
                self._roster[day].items(),
                key=lambda item: item[1][DUTY_START_TIME]
            )

            for duty_name, duty_info in duties:

                assignees = [
                    person.get_name()
                    for person in duty_info[DUTY_ASSIGNEES]
                ]

                # Always output six teacher columns
                assignees.extend([""] * (6 - len(assignees)))

                start = duty_info[DUTY_START_TIME]
                end = duty_info[DUTY_END_TIME]

                duration = (
                    Person.time_to_index(end)
                    - Person.time_to_index(start)
                ) * 0.5

                rows.append([
                    day,
                    start,
                    end,
                    duration,
                    duty_name,
                    *assignees
                ])

        df = pd.DataFrame(
            rows,
            columns=[
                "Day",
                "Start",
                "End",
                "Duration (Hours)",
                "Duty",
                "Teacher 1",
                "Teacher 2",
                "Teacher 3",
                "Teacher 4",
                "Teacher 5",
                "Teacher 6",
            ]
        )

        return df

    def create_activity_breakdown_by_class(self) -> pd.DataFrame:
        """
        Create a half-hour activity breakdown sheet by class.

        Returns
        -------
        pd.DataFrame
            Columns:
                Day
                Time
                <Class columns>
        """
        def parse_minutes(time_value):
            normalized = Person.normalize_time(time_value)
            hours, minutes = divmod(int(normalized), 100)
            return hours * 60 + minutes

        def format_slot(start_minutes, end_minutes):
            def hhmm(minutes):
                hours = minutes // 60
                mins = minutes % 60
                return f"{hours:02d}{mins:02d}"
            return f"{hhmm(start_minutes)}-{hhmm(end_minutes)}"

        if not self._roster:
            return pd.DataFrame(columns=["Day", "Time"])

        classes = self._order_classes_for_activity_breakdown()
        if not classes:
            return pd.DataFrame(columns=["Day", "Time"])

        all_days = sorted(self._roster.keys())

        min_start = None
        max_end = None

        for duties in self._roster.values():
            for duty_info in duties.values():
                start_min = parse_minutes(duty_info[DUTY_START_TIME])
                end_min = parse_minutes(duty_info[DUTY_END_TIME])
                if min_start is None or start_min < min_start:
                    min_start = start_min
                if max_end is None or end_min > max_end:
                    max_end = end_min

        if min_start is None or max_end is None:
            return pd.DataFrame(columns=["Day", "Time"] + classes)

        min_start = (min_start // 30) * 30
        max_end = ((max_end + 29) // 30) * 30

        def build_activity_details(duty_name, assignees):
            if assignees:
                return f"{duty_name}:\n" + "\n".join(assignees)
            return f"{duty_name}:"

        def are_adjacent_in_order(target_list):
            if len(target_list) < 2:
                return True
            indices = [classes.index(item) for item in target_list if item in classes]
            return indices == list(range(indices[0], indices[0] + len(indices)))

        rows = []
        for day in all_days:
            for slot_start in range(min_start, max_end, 30):
                slot_end = slot_start + 30
                row = {"Day": day, "Time": format_slot(slot_start, slot_end)}
                row.update({class_name: "" for class_name in classes})

                for duty_name, duty_info in self._roster[day].items():
                    raw_class_name = duty_info.get(DUTY_CLASS, "") or ""
                    if raw_class_name.strip().lower() == "all":
                        target_classes = classes
                    else:
                        target_classes = [
                            part.strip()
                            for part in raw_class_name.split(";")
                            if part.strip()
                        ]

                    duty_start = parse_minutes(duty_info[DUTY_START_TIME])
                    duty_end = parse_minutes(duty_info[DUTY_END_TIME])
                    if duty_start >= slot_end or duty_end <= slot_start:
                        continue

                    assignees = [person.get_name() for person in duty_info[DUTY_ASSIGNEES]]
                    details = build_activity_details(duty_name, assignees)

                    for target_class in target_classes:
                        if target_class not in classes:
                            continue
                        existing = row[target_class]
                        row[target_class] = (
                            f"{existing}\n\n{details}"
                            if existing else details
                        )

                rows.append(row)

        columns = ["Day", "Time"] + classes
        return pd.DataFrame(rows, columns=columns)

    def _order_classes_for_activity_breakdown(self):
        classes = set()
        adjacency = {}
        found_all = False

        for duties in self._roster.values():
            for duty_info in duties.values():
                raw_class_name = duty_info.get(DUTY_CLASS, "") or ""
                class_name = raw_class_name.strip()
                if not class_name:
                    continue
                if class_name.lower() == "all":
                    found_all = True
                    continue

                parts = [part.strip() for part in class_name.split(";") if part.strip()]
                for part in parts:
                    classes.add(part)
                for left, right in zip(parts, parts[1:]):
                    adjacency.setdefault(left, set()).add(right)
                    adjacency.setdefault(right, set()).add(left)

        if not classes:
            return ["All"] if found_all else []

        ordered = []
        visited = set()

        def visit_component(start):
            queue = [start]
            component_order = []
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component_order.append(current)
                for neighbor in sorted(adjacency.get(current, [])):
                    if neighbor not in visited:
                        queue.append(neighbor)
            return component_order

        nodes = sorted(classes, key=lambda cls: (len(adjacency.get(cls, [])) == 0, cls))
        for node in nodes:
            if node in visited:
                continue
            ordered.extend(visit_component(node))

        return ordered

    def create_work_distribution(self) -> pd.DataFrame:
        """
        Create the Work Distribution dataframe.

        Returns
        -------
        pd.DataFrame
            Contains weekly totals together with daily work/rest
            breakdown for every staff member.
        """

        rows = []

        # Teacher list followed by temp list
        people = (
            self._teachers.get_list() +
            self._temps.get_list()
        )

        # All days that appear in the roster
        all_days = sorted(self._roster.keys())

        for person in people:

            worked = person.get_hours_worked_by_day()
            rests = person.get_rest_periods_by_day()

            row = {
                "Person": person.get_name(),
                "Work To Capacity": round(
                    person.get_work_capacity_ratio(),
                    2,
                ),
                "Hours Worked": person.get_hours_worked(),
                "Hours In School": person.get_hours_in_school(),
                "Rest Hours": sum(rests.values()),
            }

            # Daily breakdown
            for day in all_days:

                row[f"{day} Work"] = worked.get(day, 0)
                row[f"{day} Rest"] = rests.get(day, 0)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Put summary columns first
        summary_columns = [
            "Person",
            "Work To Capacity",
            "Hours Worked",
            "Rest Hours",
            "Hours In School",
        ]

        daily_columns = []

        for day in all_days:
            daily_columns.append(f"{day} Work")
            daily_columns.append(f"{day} Rest")

        return df[summary_columns + daily_columns]

    def create_teacher_timetable(self):

        timetables = {}

        for day in sorted(self._roster.keys()):

            # Build each person's schedule once
            schedules = {
                person.get_name(): person.build_daily_schedule(day)
                for person in self._people
            }

            # Build a common set of half-hour times for the day
            time_slots = sorted(
                {
                    entry["start"]
                    for schedule in schedules.values()
                    for entry in schedule
                },
                key=lambda time_value: int(time_value)
            )

            # Map each person's schedule by start time
            schedule_by_time = {
                teacher_name: {
                    entry["start"]: entry["activity"]
                    for entry in schedule
                }
                for teacher_name, schedule in schedules.items()
            }

            rows = []
            for start_time in time_slots:
                end_minutes = int(start_time[:2]) * 60 + int(start_time[2:]) + 30
                end_time = f"{end_minutes // 60:02d}{end_minutes % 60:02d}"
                row = {
                    "Time": f"{start_time} - {end_time}",
                }

                for teacher_name in schedules:
                    row[teacher_name] = schedule_by_time[teacher_name].get(start_time, "")

                rows.append(row)

            timetables[day] = pd.DataFrame(rows)

        return timetables

    def _format_workbook(self, writer):

        workbook = writer.book

        header = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "#D9EAD3",
            "border": 1,
        })

        centre = workbook.add_format({
            "align": "center",
        })

        wrap_center = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })

        for sheet_name, worksheet in writer.sheets.items():

            worksheet.freeze_panes(1, 1)

            worksheet.set_row(0, 22, header)

            # Auto-size columns
            if sheet_name == "Duty Roster":
                dataframe = self.create_duty_roster()

            elif sheet_name == "Activity Breakdown":
                dataframe = self.create_activity_breakdown_by_class()

            elif sheet_name == "Work Distribution":
                dataframe = self.create_work_distribution()

            else:
                dataframe = self._timetable_sheets.get(sheet_name)

            if dataframe is not None:
                self._auto_size_columns(
                    worksheet,
                    dataframe,
                )

                if sheet_name == "Activity Breakdown":
                    self._merge_activity_breakdown_cells(
                        workbook,
                        worksheet,
                        dataframe,
                    )
                    for row_num in range(1, len(dataframe) + 1):
                        worksheet.set_row(row_num, 30)
                    worksheet.set_column(
                        0,
                        len(dataframe.columns),
                        None,
                        wrap_center,
                    )
                else:
                    worksheet.set_column(
                        0,
                        len(dataframe.columns),
                        None,
                        centre,
                    )

    def _auto_size_columns(self, worksheet, dataframe):
        for column_index, column_name in enumerate(dataframe.columns):
            max_width = max(
                len(str(column_name)),
                dataframe[column_name].astype(str).map(len).max() if len(dataframe) > 0 else 0,
            ) + 2
            worksheet.set_column(column_index, column_index, max_width)

    def _merge_activity_breakdown_cells(self, workbook, worksheet, dataframe):
        merge_format = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        # Merge equal adjacent cells for the same time row
        for row_index in range(1, len(dataframe) + 1):
            current_value = None
            merge_start = None
            for col_index in range(2, len(dataframe.columns)):
                value = str(dataframe.iat[row_index - 1, col_index])
                if value and value == current_value:
                    if merge_start is None:
                        merge_start = col_index - 1
                    end_col = col_index
                else:
                    if merge_start is not None:
                        worksheet.merge_range(
                            row_index,
                            merge_start,
                            row_index,
                            end_col,
                            current_value,
                            merge_format,
                        )
                        merge_start = None
                    current_value = value
                    end_col = col_index
            if merge_start is not None:
                worksheet.merge_range(
                    row_index,
                    merge_start,
                    row_index,
                    end_col,
                    current_value,
                    merge_format,
                )

    def _create_formats(self, workbook):
        return {
            "Duty": workbook.add_format({
                "bg_color": "#C6EFCE",
                "align": "center",
            }),

            "Rest": workbook.add_format({
                "bg_color": "#FFF2CC",
                "align": "center",
            }),

            "Empty": workbook.add_format({
                "bg_color": "#E7E6E6",
                "align": "center",
            }),
        }
