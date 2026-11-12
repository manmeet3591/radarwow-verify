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


def test_gross_error_qc():
    import pandas as pd

    from radarwow.verify import basic_qc, gross_error_qc

    t = pd.Timestamp("2022-05-05 07:00")
    g = pd.DataFrame(dict(station=["A", "B", "C", "D", "E"], time=t, gauge_mm=[46.0, 4.0, 2.0, 8.0, 0.0], radar=[3.7, 4.2, 2.1, 8.5, 12.0]))
    out = gross_error_qc(basic_qc(g, radar_col="radar"), radar_col="radar")
    assert out.set_index("station").qc.to_dict() == {"A": "gross error vs bias-scaled radar", "B": "ok", "C": "ok", "D": "ok",
                                                     "E": "gauge dry under radar rain"}
