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
