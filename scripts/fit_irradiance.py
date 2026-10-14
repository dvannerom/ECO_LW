"""Fit per-channel radiance-to-BT power law coefficients (radiance = (a*BT)**b).

For each GOES ABI LW channel, the Planck blackbody spectral radiance is
integrated over 4-20 um weighted by the channel's spectral response function
(SRF), for a range of blackbody temperatures. A power law is then fit to the
resulting (temperature, band-integrated radiance) curve. This a/b pair is the
sensor-emulation model used by scripts/compute_temperature_SBDART.py to turn
simulated SBDART nadir radiances into an effective narrowband BT; it is a
distinct approximation from the operational per-file Planck-coefficient BT
conversion in src/radiometry.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from spectral_response import CHANNELS, load_goes_filters

from channel_emulation import band_radiance, power_law, fit_channel_power_law


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--filter-dir", type=Path, default=ROOT / "data/inputs/goes_channels",
	                     help="Directory holding per-channel GOES ABI spectral response files")
	parser.add_argument("--t-min", type=float, default=200.0, help="Lowest fitted temperature, K")
	parser.add_argument("--t-max", type=float, default=320.0, help="Highest fitted temperature, K")
	parser.add_argument("--t-step", type=float, default=1.0, help="Temperature grid step, K")
	parser.add_argument("--output", type=Path,
	                     default=ROOT / "data/abi_proxy/models/n2b/channel_radiance_power_law.json",
	                     help="Output JSON path for fitted a/b coefficients")
	args = parser.parse_args()

	temperatures = np.arange(args.t_min, args.t_max + args.t_step, args.t_step)
	lambdas_goes, response_goes = load_goes_filters(args.filter_dir, CHANNELS)

	a_coeffs, b_coeffs = [], []
	for channel, lambdas_nm, response in zip(CHANNELS, lambdas_goes, response_goes):
		a, b, radiance = fit_channel_power_law(lambdas_nm, response, temperatures)
		print(f"{channel}: a={a:.8e}, b={b:.8f}")
		a_coeffs.append(a)
		b_coeffs.append(b)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w") as handle:
		json.dump({"channels": CHANNELS, "a": a_coeffs, "b": b_coeffs}, handle, indent=2)
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()
