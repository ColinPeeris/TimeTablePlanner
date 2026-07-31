# TimeTablePlanner User Guide

This document explains how to use the timetable planner for people who are not technical.
It describes the files you need, what each file should contain, and how to run the planner.

## What this tool does

The planner takes two Excel files as input:

1. `DutiesBreakdown.xlsx`
2. `AvailabilityList.xlsx`

It uses these files to assign available teachers and temporary staff to daily duties.
The final schedule is written to `teacher_schedule_with_duties.xlsx`.

## What you need to provide

### 1. DutiesBreakdown.xlsx

This file defines the daily activities.
For now, the planner assumes that every day has the same set of activities.

The file must include these columns:

- `Activity`
  - Name of the duty or task (for example: "Hall Duty", "Bus Supervision").
- `Session`
  - Either `AM` or `PM`.
- `Start Time`
  - The start time in `HHMM` format, such as `0900` for 9:00 AM.
- `End Time`
  - The end time in `HHMM` format, such as `1400` for 2:00 PM.
- `Minimum Requirement`
  - The minimum number of staff needed for that duty.
- `Ideal Case`
  - The ideal number of staff for that duty.

Example row:

| Activity       | Session | Start Time | End Time | Minimum Requirement | Ideal Case |
|----------------|---------|------------|----------|---------------------|------------|
| Hall Duty      | AM      | 0900       | 1200     | 1                   | 2          |
| Playground     | PM      | 1400       | 1800     | 2                   | 3          |

### 2. AvailabilityList.xlsx

This file lists when teachers and temporary staff are available to work.

It must contain two sheets:

- `Teachers`
- `Temps`

Each sheet should contain one row for each staff member's availability on each day.

The file must include the following columns:

- `Day`
  - The day of the week (for example: Monday, Tuesday).
- `Date`
  - The calendar date.
- `Staff Name`
  - The name of the teacher or temporary staff member.
- `Start Time`
  - The time the staff member starts work, in `HHMM` format (for example: `0900`).
- `End Time`
  - The time the staff member finishes work, in `HHMM` format (for example: `1400` or `1800`).

Example layout:

| Day | Date | Staff Name | Start Time | End Time |
|-----|------|------------|------------|----------|
| Monday | 18/11/2024 | Denise | 0900 | 1800 |
| Monday | 18/11/2024 | Melissa | 0900 | 1400 |
| Monday | 18/11/2024 | Lay Bee | 1300 | 1800 |
| Tuesday | 19/11/2024 | Denise | 0900 | 1800 |

Notes:

- Each row represents the availability of one staff member for one day.
- Times must be entered in `HHMM` format.
- If a staff member is unavailable on a particular day, simply omit that row.
- Multiple rows may be used for the same staff member on the same day if they have more than one working period.

## How to run the planner

From the project folder, run:

```powershell
(venv) PS C:\Colin\PythonProjects\TimeTablePlanner> python .\main.py
```

When the run finishes, you should see this message:

```text
Data has been written to teacher_schedule_with_duties.xlsx
```

## What you get after running

The planner creates `teacher_schedule_with_duties.xlsx` containing:

- A sheet with the duty roster for each day and the assigned staff.
- A sheet showing how much work each person was assigned.

## Tips

- Keep `DutiesBreakdown.xlsx` and `AvailabilityList.xlsx` in the same folder as `main.py`.
- If you change the duty list, keep the sheet structure the same.
- If you change availability, make sure each staff name is entered consistently.

## Summary

To use the planner:

1. Create or update `DutiesBreakdown.xlsx` with your daily duties.
2. Create or update `AvailabilityList.xlsx` with teacher and temp availability.
3. Run `python .\main.py`.
4. Open `teacher_schedule_with_duties.xlsx` to review the final schedule.
