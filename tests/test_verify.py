import numpy as np
import xarray as xr

from radarwow.verify import gate_latlon, scores, to_mrms_grid


def test_gate_positions_are_at_the_right_distance():
    lat, lon, s = gate_latlon(np.array([100e3]), np.array([0.0, 90.0]), 0.5, 35.0, -97.0)
    assert np.allclose(s, s[0, 0]) and abs(s[0, 0] / 1e3 - 100) < 0.5
    assert abs((lat[0, 0] - 35.0) * 111.2 - s[0, 0] / 1e3) < 0.5 and abs(lon[1, 0] + 97.0) > 1.0


def test_gridding_averages_gates_in_a_cell():
    ref = xr.DataArray(np.full((3, 3), 2.0), dims=("latitude", "longitude"),
                       coords={"latitude": [35.02, 35.01, 35.0], "longitude": [-97.0, -96.99, -96.98]})
    lat = np.array([[35.0, 35.0]])
    lon = np.array([[-97.0, -97.0]])
    ours, theirs, _ = to_mrms_grid(np.array([[1.0, 3.0]]), lat, lon, ref)
    assert ours.tolist() == [2.0] and theirs.tolist() == [2.0]


def test_scores_perfect():
    x = np.linspace(0, 10, 50)
    sc = scores(x, x)
    assert np.isclose(sc["bias_ratio"], 1) and np.isclose(sc["corr"], 1) and sc["rmse"] == 0
