"""Independent environmental references from numerical models (public AWS archives, no account).

HRRR (High-Resolution Rapid Refresh, 3 km, CONUS) analyses are on AWS (bucket ``noaa-hrrr-bdp-pds``)
since 2014. We read single GRIB2 messages with HTTP range requests using the ``.idx`` inventory, so a
field costs ~1 MB instead of a 100+ MB file.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr


def hrrr_field(time, match: str, cache_dir: str) -> xr.DataArray:
    """One HRRR analysis (f00, surface file) message whose inventory line contains ``match``,
    e.g. ``":HGT:0C isotherm:"`` for the height of the 0 C isotherm (m above mean sea level)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    t = pd.Timestamp(time)
    key = f"hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf00.grib2"
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"hrrr_{t:%Y%m%d%H}_{''.join(c for c in match if c.isalnum())}.grib2")
    if not os.path.exists(local):
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        idx = s3.get_object(Bucket="noaa-hrrr-bdp-pds", Key=key + ".idx")["Body"].read().decode().splitlines()
        for i, line in enumerate(idx):
            if match in line:
                start = int(line.split(":")[1])
                end = int(idx[i + 1].split(":")[1]) - 1 if i + 1 < len(idx) else None
                break
        else:
            raise KeyError(f"{match} not in {key}")
        rng = f"bytes={start}-{end}" if end else f"bytes={start}-"
        with open(local, "wb") as f:
            f.write(s3.get_object(Bucket="noaa-hrrr-bdp-pds", Key=key, Range=rng)["Body"].read())
    ds = xr.open_dataset(local, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]].load()
    return da.assign_coords(longitude=((da["longitude"] + 180) % 360) - 180)


def hrrr_freezing_level(time, lat, lon, cache_dir: str, radius_km: float = 0.0) -> float:
    """HRRR 0 C isotherm height (m MSL) at (lat, lon); with ``radius_km`` > 0 the median over that radius."""
    da = hrrr_field(time, ":HGT:0C isotherm:", cache_dir)
    la, lo = da["latitude"].values, da["longitude"].values
    d_km = 111.2 * np.sqrt((la - lat) ** 2 + ((lo - lon) * np.cos(np.deg2rad(lat))) ** 2)
    if radius_km > 0:
        return float(np.nanmedian(da.values[d_km <= radius_km]))
    return float(da.values[np.unravel_index(np.argmin(d_km), d_km.shape)])


def t2m_freezing_level(time, lat, lon, cache_dir: str, lapse_k_per_km: float = 6.5) -> float:
    """Rough 0 C height (m MSL) outside the HRRR domain: ground height + 2 m temperature / lapse rate, from the
    ERA5-based Open-Meteo archive (free, no key). A standard-atmosphere assumption; use only as a prior for the
    radar's own melting-layer detection."""
    import json
    import urllib.request

    t = pd.Timestamp(time)
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"openmeteo_t2m_{lat:.2f}_{lon:.2f}_{t:%Y%m%d}.json")
    if not os.path.exists(local):
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat:.3f}&longitude={lon:.3f}"
               f"&start_date={t:%Y-%m-%d}&end_date={t:%Y-%m-%d}&hourly=temperature_2m")
        with open(local, "w") as f:
            f.write(urllib.request.urlopen(url, timeout=60).read().decode())
    d = json.load(open(local))
    temps = d["hourly"]["temperature_2m"]
    t2m = float(temps[min(t.hour, len(temps) - 1)])
    return float(d.get("elevation", 0.0)) + max(t2m, 0.0) / lapse_k_per_km * 1e3


def pressure_level_freezing_level(time, lat, lon, cache_dir: str, fallback=True) -> float:
    """0 C height (m MSL) interpolated between pressure levels (1000-300 hPa temperature and geopotential height) from
    the Open-Meteo historical-forecast archive (free, no key; global models from 2022). This sees the free atmosphere,
    unlike `t2m_freezing_level`, which on a mountain-top radar at night is far too low (Tablazo, Colombia: 2 m
    temperature put 0 C 0.6-1.2 km above the antenna, the pressure levels 1.6 km). Where the archive has no data
    (it returned none for Finland), falls back to `t2m_freezing_level` if ``fallback``."""
    import json
    import urllib.request

    t = pd.Timestamp(time)
    levels = [1000, 925, 850, 700, 600, 500, 400, 300]
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"openmeteo_pl_{lat:.2f}_{lon:.2f}_{t:%Y%m%d}.json")
    if not os.path.exists(local):
        var = ",".join(f"temperature_{p}hPa,geopotential_height_{p}hPa" for p in levels)
        url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat:.3f}&longitude={lon:.3f}"
               f"&start_date={t:%Y-%m-%d}&end_date={t:%Y-%m-%d}&hourly={var}")
        with open(local, "w") as f:
            f.write(urllib.request.urlopen(url, timeout=60).read().decode())
    h = json.load(open(local)).get("hourly", {})
    i = min(t.hour, len(h.get("time", [None])) - 1)
    pts = [(h.get(f"geopotential_height_{p}hPa", [None])[i], h.get(f"temperature_{p}hPa", [None])[i]) for p in levels] if h else []
    pts = [(z, tc) for z, tc in pts if z is not None and tc is not None]
    for (z1, t1), (z2, t2) in zip(pts, pts[1:]):
        if t1 >= 0 > t2:
            return float(z1 + (z2 - z1) * t1 / (t1 - t2))
    if fallback:
        return t2m_freezing_level(time, lat, lon, cache_dir)
    raise ValueError("no 0 C crossing in the pressure-level profile")
