import argparse
import os
import sys
import traceback

from planner.scheduler import Scheduler


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the timetable planner GUI or headlessly.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the scheduler without opening the GUI.",
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Folder containing AvailabilityList.xlsx, DutiesBreakdown.xlsx, StaffAttributes.xlsx, and config.ini.",
    )
    return parser.parse_args()


def main():
    """Run the timetable planner GUI or the headless scheduler."""
    args = _parse_args()
    if not args.headless:
        from gui import run_gui

        run_gui()
        return

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(input_dir, "config.ini")
    try:
        Scheduler(
            input_dir=input_dir,
            output_dir=output_dir,
            config_file=config_file if os.path.isfile(config_file) else None,
        )
        print(f"Schedule generated successfully in {output_dir}")
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

