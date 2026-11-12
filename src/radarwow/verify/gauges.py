"""Hourly rain gauges from the NOAA Hydrometeorological Automated Data System (HADS), via the Iowa
Environmental Mesonet archive (https://mesonet.agron.iastate.edu; free, no account).

HADS relays SHEF reports from thousands of gauge networks. ``PPH*`` codes are hourly precipitation increments (inches)
for the hour ending at the report time; where a station reports both, the ``PPHRG`` series is preferred.

The data come with little quality control. `basic_qc` drops what is physically impossible (negative or > 100 mm in
an hour) and, if a radar total at the gauge is given, gross radar-gauge disagreements in the spirit of the MRMS
gauge QC (Qi et al. 2016, J. Hydrometeor. 17, 1675-1691, doi:10.1175/JHM-D-15-0188.1): a gauge reporting 0 under
>= ``dry_radar_mm`` of radar rain, or >= ``dry_radar_mm`` under no radar echo. Thresholds are radar-wow choices.

`gross_error_qc` adds a stricter check: a gauge-hour is flagged where gauge and radar differ by more than
``min_diff_mm`` and by more than a ``factor``, after the radar is scaled by that hour's median gauge/radar ratio (so a
systematic radar bias does not reject good gauges); a station with ``max_bad_hours`` or more flagged hours is
dropped. On KTLX this catches a gauge reporting 46 mm/h under 4 mm/h of radar and MRMS rain, one reporting an
hour late, and gauges stuck at 0 under 9-13 mm/h. Because it uses the radar, it leans the comparison towards the
radar; radarwow-merging notebook 01 shows results with and without it.
"""

from __future__ import annotations

import io
import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

IEM = "https://mesonet.agron.iastate.edu"


def _get(url, tries: int = 6, pause_s: float = 5.0) -> str:
    """GET with a pause between requests and back-off on HTTP 429 (the IEM archive rate-limits)."""
    import time
    import urllib.error

    for k in range(tries):
        try:
            txt = urllib.request.urlopen(url, timeout=600).read().decode()
            time.sleep(pause_s)
            return txt
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(pause_s * 2 ** (k + 1))
    raise RuntimeError("unreachable")


def hads_stations(network: str, lat0: float, lon0: float, radius_km: float) -> pd.DataFrame:
    """HADS stations of an IEM network (e.g. ``"OK_DCP"``, ``"CA_DCP"``) within ``radius_km``."""
    d = json.load(urllib.request.urlopen(f"{IEM}/geojson/network/{network}.geojson", timeout=60))
    rows = []
    for f in d["features"]:
        lo, la = f["geometry"]["coordinates"][:2]
        dist = math.hypot((la - lat0) * 111.2, (lo - lon0) * 111.2 * math.cos(math.radians(lat0)))
        if dist <= radius_km:
            rows.append(dict(station=f["properties"]["sid"], lat=la, lon=lo, name=f["properties"].get("sname", "")))
    return pd.DataFrame(rows)


def fetch_hads(networks, lat0: float, lon0: float, radius_km: float, start, end, cache_dir: str) -> pd.DataFrame:
    """Hourly gauge rain (mm) for the hours ending in (start, end], one row per station and hour.

    Columns: station, lat, lon, time (start of the hour, to match radar totals), gauge_mm.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    networks = [networks] if isinstance(networks, str) else list(networks)
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"hads_{'_'.join(networks)}_{lat0:.2f}_{lon0:.2f}_{radius_km:.0f}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv")
    if os.path.exists(local):
        return pd.read_csv(local, parse_dates=["time"])
    out = []
    for net in networks:
        st = hads_stations(net, lat0, lon0, radius_km)
        if st.empty:
            continue
        parts = []
        for i in range(0, len(st), 150):  # long station lists exceed the server's URL limit
            q = [("network", net), ("sts", f"{start:%Y-%m-%dT%H:%MZ}"), ("ets", f"{end:%Y-%m-%dT%H:%MZ}")] + [("stations", s) for s in st.station[i:i + 150]]
            txt = _get(f"{IEM}/cgi-bin/request/hads.py?" + urllib.parse.urlencode(q))
            parts.append(pd.read_csv(io.StringIO(txt), na_values=["M", ""], low_memory=False))
        df = pd.concat(parts, ignore_index=True)
        cols = sorted((c for c in df.columns if c.startswith("PPH")), key=lambda c: (not c.startswith("PPHRG"), c))
        if not cols:
            continue
        df["utc_valid"] = pd.to_datetime(df["utc_valid"])
        df = df[df["utc_valid"].dt.minute == 0]  # hourly increments ending on the hour
        val = df[cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]  # preferred code first
        g = pd.DataFrame({"station": df["station"], "time": df["utc_valid"] - pd.Timedelta(hours=1), "gauge_mm": val * 25.4}).dropna()
        g = g.groupby(["station", "time"], as_index=False)["gauge_mm"].first()
        out.append(g.merge(st[["station", "lat", "lon"]], on="station"))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["station", "time", "gauge_mm", "lat", "lon"])
    res = res[(res["time"] >= start) & (res["time"] < end)]
    res.to_csv(local, index=False)
    return res


def basic_qc(g: pd.DataFrame, radar_col: str | None = None, max_mm: float = 100.0, dry_radar_mm: float = 3.0) -> pd.DataFrame:
    """Drop impossible values and, with ``radar_col``, gross radar-gauge disagreements; adds a ``qc`` reason column."""
    g = g.copy()
    g["qc"] = "ok"
    g.loc[(g["gauge_mm"] < 0) | (g["gauge_mm"] > max_mm), "qc"] = "impossible value"
    if radar_col is not None:
        r = g[radar_col]
        g.loc[(g["qc"] == "ok") & (g["gauge_mm"] == 0) & (r >= dry_radar_mm), "qc"] = "gauge dry under radar rain"
        g.loc[(g["qc"] == "ok") & (g["gauge_mm"] >= dry_radar_mm) & (r == 0), "qc"] = "gauge rain with no radar echo"
        g.loc[(g["qc"] == "ok") & ~np.isfinite(r), "qc"] = "no radar value"
    return g


def gross_error_qc(g: pd.DataFrame, radar_col: str, factor: float = 3.0, min_diff_mm: float = 5.0, max_bad_hours: int = 2) -> pd.DataFrame:
    """Flag gross gauge-radar disagreements (after scaling the radar by the hourly median gauge/radar ratio); see module doc.

    Expects the ``qc`` column from `basic_qc`; only gauge-hours still marked "ok" are tested or changed.
    """
    g = g.copy()
    ok = g["qc"] == "ok"
    wet = ok & (g["gauge_mm"] >= 1.0) & (g[radar_col] >= 1.0)
    # robust hourly bias: median gauge/radar ratio (a single bad gauge would distort a sum ratio)
    scale = (g.loc[wet, "gauge_mm"] / g.loc[wet, radar_col]).groupby(g.loc[wet, "time"]).median()
    r = g[radar_col] * g["time"].map(scale).fillna(1.0)
    gg = g["gauge_mm"]
    ratio = np.maximum((gg + 0.5) / (r + 0.5), (r + 0.5) / (gg + 0.5))
    bad = ok & ((gg - r).abs() > min_diff_mm) & (ratio > factor)
    g.loc[bad, "qc"] = "gross error vs bias-scaled radar"
    nbad = bad.groupby(g["station"]).sum()
    drop = nbad[nbad >= max_bad_hours].index
    g.loc[g["station"].isin(drop) & (g["qc"] == "ok"), "qc"] = "station with repeated gross errors"
    return g
