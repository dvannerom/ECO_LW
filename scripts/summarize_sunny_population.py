"""Export numerical Sunny population diagnostics independently of plotting."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scene_population import write_population_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    write_population_summary(args.input, args.summary, args.csv, args.report)


if __name__ == "__main__":
    main()
