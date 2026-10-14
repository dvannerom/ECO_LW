"""Command-line entry point for shared model preparation and nominal assessment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nominal_processing import main


if __name__ == "__main__":
    main()
