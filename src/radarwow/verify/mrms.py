"""Compare a single-radar accumulation with MRMS products from the public AWS bucket (no account).

MRMS (Zhang et al. 2016, BAMS, doi:10.1175/BAMS-D-14-00174.1) "MultiSensor_QPE_01H_Pass2" is radar QPE
adjusted with rain gauges: an independent reference for one radar's estimate, but not truth (it shares
radar input and has its own errors). "RadarOnly_QPE_01H" is MRMS without gauges: a peer benchmark.
Archive on AWS starts in October 2020. Grid: 0.01 deg lat/lon, values in mm, negative = missing.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd
import xarray as xr

BUCKET = "noaa-mrms-pds"
PRODUCTS = {"gauge_corrected": "MultiSensor_QPE_01H_Pass2_00.00", "radar_only": "RadarOnly_QPE_01H_00.00"}


def fetch_mrms(product: str, hour_end, cache_dir: str) -> xr.DataArray:
    """MRMS 1-hour accumulation ending at ``hour_end`` (UTC). ``product``: key of PRODUCTS or full name."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    name = PRODUCTS.get(product, product)
    t = pd.Timestamp(hour_end)
    key = f"CONUS/{name}/{t:%Y%m%d}/MRMS_{name}_{t:%Y%m%d-%H%M%S}.grib2.gz"
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(key)[:-3])
    if not os.path.exists(local):
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        with open(local, "wb") as f:
            f.write(gzip.decompress(body))
    ds = xr.open_dataset(local, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]].load()
    da = da.where(da >= 0)
    da = da.assign_coords(longitude=((da["longitude"] + 180) % 360) - 180)
    da.attrs.update(product=name, hour_end=str(t))
    return da


def gate_latlon(range_m, azimuth_deg, elevation_deg, lat0, lon0):
    """Ground position of every gate (4/3-Earth beam path, then a local spherical approximation;
    error well under a 0.01 deg MRMS cell within 150 km)."""
    re = 6371e3
    ke = 4.0 / 3.0 * re
    r = np.asarray(range_m)[None, :]
    el = np.deg2rad(elevation_deg)
    h = np.sqrt(r ** 2 + ke ** 2 + 2 * r * ke * np.sin(el)) - ke
    s = ke * np.arcsin(r * np.cos(el) / (ke + h))  # ground distance
    az = np.deg2rad(np.asarray(azimuth_deg))[:, None]
    lat = lat0 + np.rad2deg(s * np.cos(az) / re)
    lon = lon0 + np.rad2deg(s * np.sin(az) / (re * np.cos(np.deg2rad(lat0))))
    return lat, lon, np.broadcast_to(s, lat.shape)


def to_mrms_grid(values, lat, lon, ref: xr.DataArray, ground_km=None, max_km: float = 150.0):
    """Average polar values into the cells of the MRMS grid ``ref``; returns (ours, theirs, cell distance km)."""
    la, lo = ref["latitude"].values, ref["longitude"].values
    dla, dlo = la[1] - la[0], lo[1] - lo[0]
    ok = np.isfinite(values)
    if ground_km is not None:
        ok &= ground_km <= max_km
    i = np.round((lat[ok] - la[0]) / dla).astype(int)
    j = np.round((lon[ok] - lo[0]) / dlo).astype(int)
    inside = (i >= 0) & (i < la.size) & (j >= 0) & (j < lo.size)
    i, j, v = i[inside], j[inside], values[ok][inside]
    flat = i * lo.size + j
    uniq, inv = np.unique(flat, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    ours = sums / counts
    ci, cj = uniq // lo.size, uniq % lo.size
    theirs = ref.values[ci, cj]
    d = ground_km[ok][inside] if ground_km is not None else None
    dist = np.bincount(inv, weights=d) / counts if d is not None else None
    return ours, theirs, dist


def scores(est, ref, threshold: float = 0.1):
    """Continuous scores over cells where both exist and at least one exceeds ``threshold`` mm."""
    m = np.isfinite(est) & np.isfinite(ref) & ((est >= threshold) | (ref >= threshold))
    e, r = est[m], ref[m]
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    return dict(n=int(m.sum()), bias_ratio=float(e.sum() / max(r.sum(), 1e-9)), mean_error=float(np.mean(e - r)),
                rmse=float(np.sqrt(np.mean((e - r) ** 2))), corr=float(np.corrcoef(e, r)[0, 1]),
                mae=float(np.mean(np.abs(e - r))))
