"""Raw ABI source configuration preserves satellite/channel ordering."""

from pathlib import Path

import pytest

from preprocess_data_ABI import build_files


def test_configured_raw_sources_and_channel_order(tmp_path):
    sources = {
        "year": 2030,
        "primary_root": str(tmp_path / "primary"),
        "extra_channels_root": str(tmp_path / "extra"),
        "cloud_height_root": str(tmp_path / "cloud"),
    }
    expected = []
    for satellite in ("G16", "G18"):
        for channel in range(8, 17):
            root = sources["extra_channels_root" if channel in (9, 10, 13) else "primary_root"]
            path = Path(root) / satellite / f"ABI_C{channel:02d}_s2030245.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.with_name(path.name.replace("245", "246")).touch()
            path.with_name(path.name.replace("2030", "2024")).touch()
            expected.append([str(path.resolve())])
        path = Path(sources["cloud_height_root"]) / satellite / f"ACHA2KMF_{satellite}_s2030245.nc"
        path.parent.mkdir(parents=True)
        path.touch()
        expected.append([str(path.resolve())])
    assert build_files(245, sources) == tuple(expected)
    Path(expected[0][0]).unlink()
    with pytest.raises(ValueError, match="G16 C08"):
        build_files(245, sources)
