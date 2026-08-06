# Scheduler configuration
# FAIRNESS_MODE options:
#  - 'week'    : minimize weekly stddev (original behaviour)
#  - 'day_sum' : minimize sum of daily stddevs across the week
#  - 'day_max' : minimize the worst-day stddev (minimax)

FAIRNESS_MODE = "week"
VALID_FAIRNESS_MODES = ("week", "day_sum", "day_max")

# Lunch / break requirements used by Scheduler optimization.
# These settings are applied only to teachers during the specified window.
LUNCH_BREAK_START = "1130"
LUNCH_BREAK_END = "1430"
LUNCH_BREAK_MIN_REST_SLOTS = 1
