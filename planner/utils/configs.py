"""
Configuration loader for TimeTablePlanner.

This module exposes constants used by the scheduling logic. Values are
read from a `config.ini` file if present (project root or path given by
`TIMETABLE_CONFIG` env var). When the file or keys are missing sensible
defaults are used so the program still runs without a config file.

Supported sections/keys (example `config.ini`):

[fairness]
mode = week          ; one of week|day_sum|day_max

[lunch]
start = 1130         ; HHMM
end = 1400           ; HHMM
min_rest_slots = 2   ; integer number of half-hour rest slots required

"""
from __future__ import annotations

import os
from configparser import ConfigParser

# Defaults
DEFAULTS = {
	"fairness_mode": "week",
	"valid_fairness_modes": ("week", "day_sum", "day_max"),

	"lunch_start": "1130",
	"lunch_end": "1400",
	"lunch_min_rest_slots": 2,

	"schedule_start": "0700",
    "schedule_end": "1900",
    "schedule_slot_minutes": 30,
}


def _find_config_path() -> str | None:
	# Env var takes precedence
	env_path = os.environ.get("TIMETABLE_CONFIG")
	if env_path and os.path.isfile(env_path):
		return env_path

	# Project root config.ini
	repo_root = os.getcwd()
	candidate = os.path.join(repo_root, "config.ini")
	if os.path.isfile(candidate):
		return candidate

	# No config found
	return None


def _load_config() -> dict:
	cfg = ConfigParser()
	path = _find_config_path()
	values = {}
	if path:
		cfg.read(path)
	# fairness
	values["fairness_mode"] = cfg.get("fairness", "mode", fallback=DEFAULTS["fairness_mode"]).strip()
	values["valid_fairness_modes"] = DEFAULTS["valid_fairness_modes"]

	# Schedule
	values["schedule_start"] = cfg.get(
		"schedule",
		"start",
		fallback=DEFAULTS["schedule_start"]
	).strip()

	values["schedule_end"] = cfg.get(
		"schedule",
		"end",
		fallback=DEFAULTS["schedule_end"]
	).strip()

	try:
		values["schedule_slot_minutes"] = cfg.getint(
			"schedule",
			"slot_minutes",
			fallback=DEFAULTS["schedule_slot_minutes"]
		)
	except Exception:
		values["schedule_slot_minutes"] = DEFAULTS["schedule_slot_minutes"]

	# lunch
	values["lunch_start"] = cfg.get("lunch", "start", fallback=DEFAULTS["lunch_start"]).strip()
	values["lunch_end"] = cfg.get("lunch", "end", fallback=DEFAULTS["lunch_end"]).strip()
	try:
		values["lunch_min_rest_slots"] = cfg.getint("lunch", "min_rest_slots", fallback=DEFAULTS["lunch_min_rest_slots"])
	except Exception:
		values["lunch_min_rest_slots"] = DEFAULTS["lunch_min_rest_slots"]

	# Normalise time strings to HHMM
	for k in ("lunch_start", "lunch_end", "schedule_start", "schedule_end"):
		v = values.get(k, "")
		if v is None:
			v = ""
		v = str(v).zfill(4)
		values[k] = v

	return values


_CFG = _load_config()


FAIRNESS_MODE = _CFG["fairness_mode"]
VALID_FAIRNESS_MODES = _CFG["valid_fairness_modes"]

LUNCH_BREAK_START = _CFG["lunch_start"]
LUNCH_BREAK_END = _CFG["lunch_end"]
LUNCH_BREAK_MIN_REST_SLOTS = _CFG["lunch_min_rest_slots"]

SCHEDULE_START = _CFG["schedule_start"]
SCHEDULE_END = _CFG["schedule_end"]
SCHEDULE_SLOT_MINUTES = _CFG["schedule_slot_minutes"]
