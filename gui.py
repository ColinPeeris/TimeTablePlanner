"""PySimpleGUI front end for selecting, editing, and generating schedules."""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from configparser import ConfigParser
from datetime import date

import pandas as pd
import PySimpleGUI as sg

from planner.scheduler import Scheduler


CASE_NAMES = ("Thomson", "Farrer", "East Coast")
REQUIRED_FILES = (
    "AvailabilityList.xlsx",
    "config.ini",
    "DutiesBreakdown.xlsx",
    "StaffAttributes.xlsx",
)
AVAILABILITY_COLUMNS = (
    "Day",
    "Date",
    "Staff Name",
    "Start Time",
    "End Time",
    "Staff Type",
    "Expected Capacity",
)
DUTY_COLUMNS = (
    "Day",
    "Date",
    "Activity",
    "Class",
    "Session",
    "Start Time",
    "End Time",
    "Minimum Requirement",
    "Ideal Case",
    "Required Function",
    "Restricted Function",
    "Staff Preference",
)
STAFF_ATTRIBUTE_COLUMNS = (
    "Attribute Type",
    "Staff Name",
    "Function or Restriction",
)
TEMPLATE_AVAILABILITY_COLUMNS = tuple(column for column in AVAILABILITY_COLUMNS if column != "Date")
TEMPLATE_DUTY_COLUMNS = tuple(column for column in DUTY_COLUMNS if column != "Date")


def _case_path(root: str, case_name: str) -> str:
    """Return the input directory for a named Sunny Day case."""
    return os.path.join(root, "SunnyDayCase", case_name)


def _create_dated_rows(dataframe: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Expand weekday template rows into one row for each selected calendar date.

    Existing Date values are ignored because the GUI derives dates from the
    selected range and each row's Day value.
    """
    if "Day" not in dataframe.columns:
        raise ValueError("The workbook must contain a 'Day' column")

    template = dataframe.drop(columns=["Date"], errors="ignore").copy()
    template_days = template["Day"].astype(str).str.strip().str.casefold()
    dated_rows = []
    for selected_date in pd.date_range(start, end, freq="D"):
        matching_rows = template.loc[template_days == selected_date.day_name().casefold()].copy()
        if not matching_rows.empty:
            matching_rows.insert(1, "Date", selected_date.strftime("%Y-%m-%d"))
            dated_rows.append(matching_rows)

    if not dated_rows:
        columns = list(template.columns)
        columns.insert(1, "Date")
        return pd.DataFrame(columns=columns)
    return pd.concat(dated_rows, ignore_index=True)


def _table_rows(dataframe: pd.DataFrame):
    """Convert a dataframe into string rows suitable for a PySimpleGUI Table."""
    return dataframe.fillna("").astype(str).values.tolist()


def _template_dataframe(dataframe: pd.DataFrame, columns) -> pd.DataFrame:
    """Prepare a weekday-template dataframe for display in the template editor."""
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    result = dataframe.drop(columns=["Date"], errors="ignore").copy()
    result = result.reindex(columns=[column for column in columns if column in result.columns])
    result["Day"] = result["Day"].astype(str).str.strip()
    result["_day_order"] = pd.Categorical(result["Day"], categories=ordered_days, ordered=True)
    return result.sort_values("_day_order", kind="stable").drop(columns="_day_order").reset_index(drop=True).astype(object)


def _read_config_rows(case_dir: str) -> pd.DataFrame:
    """Read config.ini as editable section, key, and value rows."""
    config = ConfigParser()
    config.read(os.path.join(case_dir, "config.ini"))
    rows = []
    for section in config.sections():
        for key, value in config.items(section):
            rows.append({"Section": section, "Key": key, "Value": value})
    return pd.DataFrame(rows, columns=["Section", "Key", "Value"])


def _read_staff_attributes(case_dir: str) -> pd.DataFrame:
    """Load required functions and restrictions into one editable dataframe."""
    workbook = os.path.join(case_dir, "StaffAttributes.xlsx")
    required = pd.read_excel(workbook, sheet_name="Special Functions")
    required = required.rename(columns={"Special Function": "Function or Restriction"})
    required.insert(0, "Attribute Type", "Required Function")
    restricted = pd.read_excel(workbook, sheet_name="Restrictions")
    restricted = restricted.rename(columns={"Restrictions": "Function or Restriction"})
    restricted.insert(0, "Attribute Type", "Restriction")
    return pd.concat(
        [required[list(STAFF_ATTRIBUTE_COLUMNS)], restricted[list(STAFF_ATTRIBUTE_COLUMNS)]],
        ignore_index=True,
    )


def _read_runtime_config(case_dir: str) -> dict:
    """Read runtime scheduling settings from a case config.ini file."""
    config = ConfigParser()
    config.read(os.path.join(case_dir, "config.ini"))
    return {
        "fairness": config.get("fairness", "mode", fallback="week").strip(),
        "lunch_start": config.get("lunch", "start", fallback="1130").strip(),
        "lunch_end": config.get("lunch", "end", fallback="1400").strip(),
        "lunch_min_rest_slots": config.getint("lunch", "min_rest_slots", fallback=2),
    }


def _set_runtime_config_controls(window, case_dir: str):
    """Populate the main window's runtime settings from a case configuration."""
    settings = _read_runtime_config(case_dir)
    window["FAIRNESS"].update(value=settings["fairness"])
    window["LUNCH_START"].update(value=settings["lunch_start"])
    window["LUNCH_END"].update(value=settings["lunch_end"])
    window["LUNCH_REST"].update(value=str(settings["lunch_min_rest_slots"]))


def _edit_dialog(title, columns, row):
    """Show a modal row editor and return edited values, or None on cancellation."""
    layout = []
    for column, value in zip(columns, row):
        display_value = "" if pd.isna(value) else str(value)
        layout.append([sg.Text(column, size=(22, 1)), sg.Input(display_value, key=column)])
    layout.append([sg.Button("Save"), sg.Button("Cancel")])
    window = sg.Window(title, layout, modal=True)
    event, values = window.read()
    window.close()
    return [values[column] for column in columns] if event == "Save" else None


def _show_error(error):
    """Print a full traceback and display the exception in a GUI popup."""
    print("\n[TimeTablePlanner ERROR]", flush=True)
    traceback.print_exception(type(error), error, error.__traceback__)
    sg.popup_error(str(error))


def _write_template_files(case_dir, availability, duties, staff_attributes, config_rows):
    """Persist explicitly confirmed template-editor changes to a case folder."""
    availability.to_excel(os.path.join(case_dir, "AvailabilityList.xlsx"), index=False)
    duties.to_excel(os.path.join(case_dir, "DutiesBreakdown.xlsx"), index=False)

    required = staff_attributes.loc[
        staff_attributes["Attribute Type"] == "Required Function",
        ["Staff Name", "Function or Restriction"],
    ].rename(columns={"Function or Restriction": "Special Function"})
    restrictions = staff_attributes.loc[
        staff_attributes["Attribute Type"] == "Restriction",
        ["Staff Name", "Function or Restriction"],
    ].rename(columns={"Function or Restriction": "Restrictions"})
    with pd.ExcelWriter(os.path.join(case_dir, "StaffAttributes.xlsx"), engine="openpyxl") as writer:
        required.to_excel(writer, sheet_name="Special Functions", index=False)
        restrictions.to_excel(writer, sheet_name="Restrictions", index=False)

    config = ConfigParser()
    for row in config_rows.itertuples(index=False):
        section = str(row.Section).strip()
        key = str(row.Key).strip()
        if not section or not key:
            continue
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key, str(row.Value))
    with open(os.path.join(case_dir, "config.ini"), "w", encoding="utf-8") as config_file:
        config.write(config_file)


def run_template_editor(base_dir):
    """Open the editor for weekday templates, attributes, and config.ini.

    Files under the selected case are written only when the user confirms the
    Save Template Changes action. Returns the selected case name when closed.
    """
    layout = [
        [sg.Text("Template case"), sg.Combo(CASE_NAMES, default_value=CASE_NAMES[0], readonly=True, key="T_CASE")],
        [sg.Button("Load Templates", key="T_LOAD"), sg.Button("Validate Template", key="T_VALIDATE", disabled=True), sg.Button("Save Template Changes", key="T_SAVE", disabled=True), sg.Text("", key="T_STATUS", size=(65, 1))],
        [sg.Text("Availability (weekday templates; dates are hidden)")],
        [sg.Table([], headings=TEMPLATE_AVAILABILITY_COLUMNS, key="T_AVAIL", num_rows=7, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[14] * len(TEMPLATE_AVAILABILITY_COLUMNS))],
        [sg.Button("Edit Selected Availability", key="T_EDIT_AVAIL", disabled=True), sg.Button("Delete Selected Availability", key="T_DELETE_AVAIL", disabled=True)],
        [sg.Text("Duties (weekday templates; dates are hidden)")],
        [sg.Table([], headings=TEMPLATE_DUTY_COLUMNS, key="T_DUTIES", num_rows=7, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[14] * len(TEMPLATE_DUTY_COLUMNS))],
        [sg.Button("Edit Selected Duty", key="T_EDIT_DUTY", disabled=True), sg.Button("Delete Selected Duty", key="T_DELETE_DUTY", disabled=True)],
        [sg.Text("Staff Attributes")],
        [sg.Table([], headings=STAFF_ATTRIBUTE_COLUMNS, key="T_ATTRIBUTES", num_rows=6, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[20, 20, 30])],
        [sg.Button("Edit Selected Staff Attribute", key="T_EDIT_ATTRIBUTE", disabled=True), sg.Button("Delete Selected Staff Attribute", key="T_DELETE_ATTRIBUTE", disabled=True)],
        [sg.Text("config.ini")],
        [sg.Table([], headings=("Section", "Key", "Value"), key="T_CONFIG", num_rows=6, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[18, 24, 45])],
        [sg.Button("Edit Selected Config", key="T_EDIT_CONFIG", disabled=True), sg.Button("Delete Selected Config", key="T_DELETE_CONFIG", disabled=True)],
    ]
    window = sg.Window("Time Table Planner - Template Editor", layout, resizable=True, finalize=True)
    availability = duties = staff_attributes = config_rows = None
    case_dir = None
    selected_case_name = CASE_NAMES[0]
    selected = {"T_AVAIL": None, "T_DUTIES": None, "T_ATTRIBUTES": None, "T_CONFIG": None}

    def select_row(table_key, values):
        """Track a table selection and enable its edit/delete controls."""
        selected[table_key] = values[table_key][0] if values[table_key] else None
        suffix = {"T_AVAIL": "AVAIL", "T_DUTIES": "DUTY", "T_ATTRIBUTES": "ATTRIBUTE", "T_CONFIG": "CONFIG"}[table_key]
        window[f"T_EDIT_{suffix}"].update(disabled=selected[table_key] is None)
        window[f"T_DELETE_{suffix}"].update(disabled=selected[table_key] is None)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "T_LOAD":
            try:
                selected_case_name = values["T_CASE"]
                case_dir = _case_path(base_dir, values["T_CASE"])
                missing = [name for name in REQUIRED_FILES if not os.path.exists(os.path.join(case_dir, name))]
                if missing:
                    raise FileNotFoundError(f"Missing in {case_dir}: {', '.join(missing)}")
                sheets = pd.read_excel(os.path.join(case_dir, "AvailabilityList.xlsx"), sheet_name=None)
                availability = _template_dataframe(pd.concat(sheets.values(), ignore_index=True), TEMPLATE_AVAILABILITY_COLUMNS)
                duties = _template_dataframe(pd.read_excel(os.path.join(case_dir, "DutiesBreakdown.xlsx")), TEMPLATE_DUTY_COLUMNS)
                staff_attributes = _read_staff_attributes(case_dir)
                config_rows = _read_config_rows(case_dir)
                window["T_AVAIL"].update(values=_table_rows(availability))
                window["T_DUTIES"].update(values=_table_rows(duties))
                window["T_ATTRIBUTES"].update(values=_table_rows(staff_attributes))
                window["T_CONFIG"].update(values=_table_rows(config_rows))
                window["T_VALIDATE"].update(disabled=False)
                window["T_SAVE"].update(disabled=False)
                window["T_STATUS"].update(value="Loaded templates. Changes are saved only when Save Template Changes is clicked.")
            except Exception as error:
                _show_error(error)
        elif event in selected:
            select_row(event, values)
        elif event.startswith("T_EDIT_"):
            table_key = {"T_EDIT_AVAIL": "T_AVAIL", "T_EDIT_DUTY": "T_DUTIES", "T_EDIT_ATTRIBUTE": "T_ATTRIBUTES", "T_EDIT_CONFIG": "T_CONFIG"}[event]
            dataframes = {"T_AVAIL": availability, "T_DUTIES": duties, "T_ATTRIBUTES": staff_attributes, "T_CONFIG": config_rows}
            dataframe = dataframes[table_key]
            index = selected[table_key]
            if dataframe is not None and index is not None:
                edited = _edit_dialog("Edit Template Row", list(dataframe.columns), dataframe.iloc[index].tolist())
                if edited is not None:
                    dataframe.loc[index] = edited
                    window[table_key].update(values=_table_rows(dataframe))
        elif event.startswith("T_DELETE_"):
            table_key = {"T_DELETE_AVAIL": "T_AVAIL", "T_DELETE_DUTY": "T_DUTIES", "T_DELETE_ATTRIBUTE": "T_ATTRIBUTES", "T_DELETE_CONFIG": "T_CONFIG"}[event]
            dataframes = {"T_AVAIL": availability, "T_DUTIES": duties, "T_ATTRIBUTES": staff_attributes, "T_CONFIG": config_rows}
            dataframe = dataframes[table_key]
            index = selected[table_key]
            if dataframe is not None and index is not None:
                dataframe.drop(dataframe.index[index], inplace=True)
                dataframe.reset_index(drop=True, inplace=True)
                selected[table_key] = None
                window[table_key].update(values=_table_rows(dataframe))
                window[event].update(disabled=True)
        elif event == "T_SAVE":
            try:
                if sg.popup_yes_no("Save these changes into the selected case input files?") == "Yes":
                    _write_template_files(case_dir, availability, duties, staff_attributes, config_rows)
                    window["T_STATUS"].update(value=f"Saved changes to {case_dir}")
            except Exception as error:
                _show_error(error)
        elif event == "T_VALIDATE":
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    _write_template_files(
                        temp_dir,
                        availability,
                        duties,
                        staff_attributes,
                        config_rows,
                    )
                    Scheduler(
                        input_dir=temp_dir,
                        output_dir=temp_dir,
                    )
                window["T_STATUS"].update(value="Template validation succeeded")
                sg.popup("Template validation succeeded", title="Time Table Planner")
            except Exception as error:
                window["T_STATUS"].update(value=f"Template validation failed: {error}")
                _show_error(error)
    window.close()
    return selected_case_name


def _write_edited_files(case_dir, availability, duties, staff_attributes, runtime_config, target_dir):
    """Write GUI edits to temporary scheduler inputs without changing case files."""
    config_path = os.path.join(target_dir, "config.ini")
    shutil.copy2(os.path.join(case_dir, "config.ini"), config_path)
    config = ConfigParser()
    config.read(config_path)
    if not config.has_section("fairness"):
        config.add_section("fairness")
    if not config.has_section("lunch"):
        config.add_section("lunch")
    config.set("fairness", "mode", runtime_config["fairness"])
    config.set("lunch", "start", runtime_config["lunch_start"])
    config.set("lunch", "end", runtime_config["lunch_end"])
    config.set("lunch", "min_rest_slots", runtime_config["lunch_min_rest_slots"])
    with open(config_path, "w", encoding="utf-8") as config_file:
        config.write(config_file)
    availability_path = os.path.join(target_dir, "AvailabilityList.xlsx")
    with pd.ExcelWriter(availability_path, engine="openpyxl") as writer:
        availability.to_excel(writer, sheet_name="Availability", index=False)
    staff_attributes_path = os.path.join(target_dir, "StaffAttributes.xlsx")
    required = staff_attributes.loc[
        staff_attributes["Attribute Type"] == "Required Function",
        ["Staff Name", "Function or Restriction"],
    ].rename(columns={"Function or Restriction": "Special Function"})
    restrictions = staff_attributes.loc[
        staff_attributes["Attribute Type"] == "Restriction",
        ["Staff Name", "Function or Restriction"],
    ].rename(columns={"Function or Restriction": "Restrictions"})
    with pd.ExcelWriter(staff_attributes_path, engine="openpyxl") as writer:
        required.to_excel(writer, sheet_name="Special Functions", index=False)
        restrictions.to_excel(writer, sheet_name="Restrictions", index=False)
    duties.to_excel(os.path.join(target_dir, "DutiesBreakdown.xlsx"), index=False)


def run_gui(base_dir=None):
    """Run the date-based scheduling GUI.

    The GUI reads case templates, expands weekday rows across the selected
    date range, permits runtime edits, and publishes generated output only
    after the scheduler completes successfully.
    """
    base_dir = os.path.abspath(base_dir or os.getcwd())
    # SystemDefault returns placeholder values in PySimpleGUI 6.3.0.1,
    # which Tk rejects as colors when an Input is updated.
    sg.theme("LightGrey1")
    layout = [
        [sg.Text("Sunny Day Case"), sg.Combo(CASE_NAMES, default_value=CASE_NAMES[0], readonly=True, key="CASE", enable_events=True)],
        [sg.Text("Start date (YYYY-MM-DD)"), sg.Input(key="START", size=(14, 1)), sg.CalendarButton("Select", target="START", format="%Y-%m-%d")],
        [sg.Text("End date (YYYY-MM-DD)"), sg.Input(key="END", size=(14, 1)), sg.CalendarButton("Select", target="END", format="%Y-%m-%d")],
        [sg.Text("Fairness"), sg.Combo(("week", "day_sum", "day_max"), key="FAIRNESS", readonly=True, size=(12, 1)), sg.Text("Lunch start"), sg.Input(key="LUNCH_START", size=(8, 1)), sg.Text("Lunch end"), sg.Input(key="LUNCH_END", size=(8, 1)), sg.Text("Minimum rest slots"), sg.Input(key="LUNCH_REST", size=(5, 1))],
        [sg.Button("Load Data", key="LOAD"), sg.Button("Generate Schedule", key="GENERATE", disabled=True), sg.Button("Open Template Editor", key="TEMPLATE_EDITOR"), sg.Text("", key="STATUS", size=(55, 1))],
        [sg.Text("Availability")],
        [sg.Table([], headings=AVAILABILITY_COLUMNS, key="AVAILABILITY", num_rows=8, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[14] * len(AVAILABILITY_COLUMNS))],
        [sg.Button("Edit Selected Availability", key="EDIT_AVAIL", disabled=True), sg.Button("Delete Selected Availability", key="DELETE_AVAIL", disabled=True)],
        [sg.Text("Duties")],
        [sg.Table([], headings=DUTY_COLUMNS, key="DUTIES", num_rows=8, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[14] * len(DUTY_COLUMNS))],
        [sg.Button("Edit Selected Duty", key="EDIT_DUTY", disabled=True), sg.Button("Delete Selected Duty", key="DELETE_DUTY", disabled=True)],
        [sg.Text("Staff Attributes")],
        [sg.Table([], headings=STAFF_ATTRIBUTE_COLUMNS, key="STAFF_ATTRIBUTES", num_rows=8, expand_x=True, expand_y=True, enable_events=True, auto_size_columns=False, col_widths=[20, 20, 30])],
        [sg.Button("Edit Selected Staff Attribute", key="EDIT_ATTRIBUTE", disabled=True), sg.Button("Delete Selected Staff Attribute", key="DELETE_ATTRIBUTE", disabled=True)],
    ]
    window = sg.Window("Time Table Planner", layout, resizable=True, finalize=True)
    _set_runtime_config_controls(window, _case_path(base_dir, CASE_NAMES[0]))
    availability = duties = staff_attributes = None
    case_dir = None
    selected_availability = selected_duty = selected_attribute = None

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "TEMPLATE_EDITOR":
            availability = duties = staff_attributes = None
            case_dir = None
            window["AVAILABILITY"].update(values=[])
            window["DUTIES"].update(values=[])
            window["STAFF_ATTRIBUTES"].update(values=[])
            window["GENERATE"].update(disabled=True)
            template_case = run_template_editor(base_dir)
            window["STATUS"].update(
                value=f"Template editor closed for {template_case}. Click Load Data to reload the saved files."
            )
            continue
        if event == "CASE":
            selected_case_dir = _case_path(base_dir, values["CASE"])
            if os.path.isfile(os.path.join(selected_case_dir, "config.ini")):
                _set_runtime_config_controls(window, selected_case_dir)
        if event == "LOAD":
            try:
                start = date.fromisoformat(values["START"])
                end = date.fromisoformat(values["END"])
                if start > end:
                    raise ValueError("Start date must not be after end date")
                case_dir = _case_path(base_dir, values["CASE"])
                missing = [name for name in REQUIRED_FILES if not os.path.exists(os.path.join(case_dir, name))]
                if missing:
                    raise FileNotFoundError(f"Missing in {case_dir}: {', '.join(missing)}")
                _set_runtime_config_controls(window, case_dir)
                availability_sheets = pd.read_excel(os.path.join(case_dir, "AvailabilityList.xlsx"), sheet_name=None)
                availability = pd.concat(availability_sheets.values(), ignore_index=True)
                duties = pd.read_excel(os.path.join(case_dir, "DutiesBreakdown.xlsx"))
                staff_attributes = _read_staff_attributes(case_dir)
                availability = _create_dated_rows(availability, start, end).reindex(columns=AVAILABILITY_COLUMNS).astype(object)
                duties = _create_dated_rows(duties, start, end).reindex(columns=DUTY_COLUMNS).astype(object)
                window["AVAILABILITY"].update(values=_table_rows(availability))
                window["DUTIES"].update(values=_table_rows(duties))
                window["STAFF_ATTRIBUTES"].update(values=_table_rows(staff_attributes))
                window["GENERATE"].update(disabled=False)
                if availability.empty and duties.empty:
                    window["GENERATE"].update(disabled=True)
                    window["STATUS"].update(
                        value="No workbook rows match the selected weekdays"
                    )
                else:
                    window["STATUS"].update(
                        value=(
                            f"Loaded {len(availability)} availability rows and {len(duties)} duties "
                            f"from {case_dir}"
                        )
                    )
            except Exception as error:
                _show_error(error)
        elif event == "AVAILABILITY":
            selected_availability = values["AVAILABILITY"][0] if values["AVAILABILITY"] else None
            window["EDIT_AVAIL"].update(disabled=selected_availability is None)
            window["DELETE_AVAIL"].update(disabled=selected_availability is None)
        elif event == "DUTIES":
            selected_duty = values["DUTIES"][0] if values["DUTIES"] else None
            window["EDIT_DUTY"].update(disabled=selected_duty is None)
            window["DELETE_DUTY"].update(disabled=selected_duty is None)
        elif event == "STAFF_ATTRIBUTES":
            selected_attribute = values["STAFF_ATTRIBUTES"][0] if values["STAFF_ATTRIBUTES"] else None
            window["EDIT_ATTRIBUTE"].update(disabled=selected_attribute is None)
            window["DELETE_ATTRIBUTE"].update(disabled=selected_attribute is None)
        elif event in ("EDIT_AVAIL", "EDIT_DUTY"):
            dataframe = availability if event == "EDIT_AVAIL" else duties
            index = selected_availability if event == "EDIT_AVAIL" else selected_duty
            if dataframe is not None and index is not None:
                edited = _edit_dialog("Edit row", list(dataframe.columns), dataframe.iloc[index].tolist())
                if edited is not None:
                    dataframe.loc[index] = edited
                    key = "AVAILABILITY" if event == "EDIT_AVAIL" else "DUTIES"
                    window[key].update(values=_table_rows(dataframe))
        elif event == "EDIT_ATTRIBUTE":
            if staff_attributes is not None and selected_attribute is not None:
                edited = _edit_dialog(
                    "Edit Staff Attribute",
                    list(staff_attributes.columns),
                    staff_attributes.iloc[selected_attribute].tolist(),
                )
                if edited is not None:
                    staff_attributes.loc[selected_attribute] = edited
                    window["STAFF_ATTRIBUTES"].update(values=_table_rows(staff_attributes))
        elif event in ("DELETE_AVAIL", "DELETE_DUTY", "DELETE_ATTRIBUTE"):
            if event == "DELETE_AVAIL":
                dataframe, index, key = availability, selected_availability, "AVAILABILITY"
            elif event == "DELETE_DUTY":
                dataframe, index, key = duties, selected_duty, "DUTIES"
            else:
                dataframe, index, key = staff_attributes, selected_attribute, "STAFF_ATTRIBUTES"
            if dataframe is not None and index is not None:
                dataframe.drop(dataframe.index[index], inplace=True)
                dataframe.reset_index(drop=True, inplace=True)
                window[key].update(values=_table_rows(dataframe))
                if key == "AVAILABILITY":
                    selected_availability = None
                    window["EDIT_AVAIL"].update(disabled=True)
                elif key == "DUTIES":
                    selected_duty = None
                    window["EDIT_DUTY"].update(disabled=True)
                else:
                    selected_attribute = None
                    window["EDIT_ATTRIBUTE"].update(disabled=True)
                window[event].update(disabled=True)
        elif event == "GENERATE":
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    scheduler_output_dir = os.path.join(temp_dir, "output")
                    os.makedirs(scheduler_output_dir)
                    runtime_config = {
                        "fairness": values["FAIRNESS"],
                        "lunch_start": values["LUNCH_START"],
                        "lunch_end": values["LUNCH_END"],
                        "lunch_min_rest_slots": values["LUNCH_REST"],
                    }
                    if runtime_config["fairness"] not in ("week", "day_sum", "day_max"):
                        raise ValueError("Fairness must be week, day_sum, or day_max")
                    if not runtime_config["lunch_min_rest_slots"].isdigit():
                        raise ValueError("Minimum rest slots must be a whole number")
                    _write_edited_files(case_dir, availability, duties, staff_attributes, runtime_config, temp_dir)
                    date_values = pd.concat([availability["Date"], duties["Date"]], ignore_index=True)
                    selected_dates = sorted(
                        set(
                            pd.to_datetime(
                                date_values,
                                dayfirst=False,
                                errors="raise",
                            ).dt.strftime("%Y-%m-%d")
                        )
                    )
                    Scheduler(
                        input_dir=temp_dir,
                        output_dir=scheduler_output_dir,
                        selected_dates=selected_dates,
                        config_file=os.path.join(temp_dir, "config.ini"),
                    )
                    shutil.copy2(
                        os.path.join(scheduler_output_dir, "teacher_schedule_with_duties.xlsx"),
                        base_dir,
                    )
                    shutil.copy2(
                        os.path.join(scheduler_output_dir, "teacher_schedule_with_duties.state"),
                        base_dir,
                    )
                window["STATUS"].update(value=f"Generated schedule in {base_dir}")
                sg.popup("Schedule generated successfully", title="Time Table Planner")
            except Exception as error:
                window["STATUS"].update(value=f"Generation failed: {error}")
                _show_error(error)
    window.close()


if __name__ == "__main__":
    run_gui()