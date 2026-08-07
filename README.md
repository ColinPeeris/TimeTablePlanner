# TimeTablePlanner

A timetable duty planner for assigning teachers and temps to duties based on availability.

## Structure

- `main.py` - entrypoint that starts the scheduler and writes the result to `teacher_schedule_with_duties.xlsx`
- `planner/` - package containing the refactored classes:
  - `person.py`
  - `queue.py`
  - `duty_roster.py`
  - `scheduler.py`
  - `staff_attributes.py`
- `tests/` - unit tests for the planner modules

## Requirements

The project also requires `StaffAttributes.xlsx` alongside `DutiesBreakdown.xlsx` and `AvailabilityList.xlsx`.

Use the project virtual environment before running commands.

## Run the app

From the project root:

```powershell
(venv) PS C:\Colin\PythonProjects\TimeTablePlanner> python .\main.py
Data has been written to teacher_schedule_with_duties.xlsx
```
## Configuration

The application reads an optional `config.ini` file from the project root (or from the path
pointed to by the `TIMETABLE_CONFIG` environment variable). If the file is missing, sensible
defaults are used.

Example `config.ini` (already included in the repo):

```ini
[fairness]
mode = week        ; options: week, day_sum, day_max

[lunch]
start = 1130       ; lunch window start (HHMM)
end = 1400         ; lunch window end (HHMM)
min_rest_slots = 2 ; minimum number of half-hour rest slots required during lunch
```

When packaging the application, users can customize `config.ini` to change fairness
behaviour and lunch-break requirements without modifying code.

## Scheduler state file (.state)

In addition to the Excel report, the scheduler now writes a companion state file with a
`.state` suffix (pickle-based by default). Example output:

```
teacher_schedule_with_duties.xlsx
teacher_schedule_with_duties.state
```

The `.state` file contains a `ScheduleState` object with the roster and the internal
teacher/temp queues. The file is intended for the scheduler to resume or modify
schedules programmatically (freeze days, selective rescheduling, undo, etc.). The
serialization format is isolated in `planner/state_serializer.py` so switching to JSON
in future is straightforward.

## Run unit tests

From the project root:

```powershell
(venv) PS C:\Colin\PythonProjects\TimeTablePlanner> python -m pytest -q tests
```

Expected output:

```text
.............................
29 passed in 0.72s
```

## Build executable

The project includes a `setup.sh` script that packages the application using PyInstaller.

From the project root, run:

```powershell
(venv) PS C:\Colin\PythonProjects\TimeTablePlanner> & "C:\Program Files\Git\bin\bash.exe" .\setup.sh
```

When the build completes, the distributable application will be created in the `dist/` folder.

## Notes

- Results are written to `teacher_schedule_with_duties.xlsx`.
- The `tests/` folder contains the current unit test suite for the refactored planner package.
- The executable is generated in the `dist/` directory using PyInstaller.
