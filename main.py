import sys
import traceback

from planner.scheduler import Scheduler


def main():
    """Run the timetable planner scheduler."""
    try:
        Scheduler()
        print("\nSchedule generated successfully!")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()

