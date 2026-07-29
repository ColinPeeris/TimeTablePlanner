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

This file lists which teachers and temps are available on which days.
It must contain four separate sheets:

- `Teachers_AM`
- `Teachers_PM`
- `Temps_AM`
- `Temps_PM`

Each sheet should list the day of the week, the date, and the staff members who are available for that session.

- `Teachers_AM` should contain all teachers available in the morning.
- `Teachers_PM` should contain all teachers available in the afternoon.
- `Temps_AM` should contain all temps available in the morning.
- `Temps_PM` should contain all temps available in the afternoon.

A simple example layout for a sheet:

| Day     | Date       | Staff 1 | Staff 2 | Staff 3 |
|---------|------------|---------|---------|---------|
| Monday  | 18/11/2024 | Denise  | Melissa | Lay Bee |
| Tuesday | 19/11/2024 | Denise  | Melissa | Lay Bee |

- Use the `Day` column for the day of the week.
- Use the `Date` column for the calendar date.
- List the names of available staff in additional columns.
- Empty cells are allowed when there are fewer available staff.

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
