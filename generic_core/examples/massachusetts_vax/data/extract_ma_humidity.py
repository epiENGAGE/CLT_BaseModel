"""Extract daily specific humidity averaged over Massachusetts grid cells.

Reads the gridMET specific-humidity NetCDF files (`sph_2025.nc`, `sph_2026.nc`)
in this directory, masks the CONUS grid down to cells whose centers fall
inside Massachusetts, and writes the spatial mean for each day in the
requested date range to a CSV (`date,specific_humidity`).

Usage:
    python extract_ma_humidity.py [--start 2025-08-01] [--end 2026-07-31] [--out ma_specific_humidity.csv]
"""

import argparse
import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# Massachusetts polygon (lon, lat), simplified state boundary.
# Source: PublicaMundi/MappingAPI us-states.json (public domain Census-derived shapes).
MA_POLYGON = [
    (-70.917135, 42.887974), (-70.821385, 42.331026), (-70.495457, 41.747706),
    (-70.082, 41.740176), (-70.183454, 41.394655), (-70.081858, 41.503973),
    (-69.937149, 41.706747), (-70.225407, 41.630235), (-70.291267, 41.291434),
    (-70.747965, 41.626626), (-70.804747, 41.224436), (-71.196845, 41.67751),
    (-71.319889, 41.484976), (-71.380742, 42.017087), (-71.799309, 42.006194),
    (-72.456392, 42.038326), (-73.054072, 42.039751), (-73.265024, 42.086022),
    (-73.355405, 42.301693), (-73.293972, 42.74403), (-72.532072, 42.722206),
    (-71.798405, 42.716212), (-71.085126, 42.671732), (-70.917135, 42.887974),
]


def point_in_polygon(lons, lats, polygon):
    """Vectorized ray-casting point-in-polygon test.

    `lons`/`lats` are 2D arrays of equal shape (grid cell centers);
    returns a boolean array of the same shape.
    """
    poly = np.asarray(polygon)
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    inside = np.zeros(lons.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        cond = (yi > lats) != (yj > lats)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (xj - xi) * (lats - yi) / (yj - yi) + xi
        crosses = cond & (lons < slope)
        inside ^= crosses
        j = i
    return inside


def ma_grid_mask(ds):
    """Boolean mask, shape (lat, lon), True where the grid cell center is in MA."""
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    lon_min, lon_max = min(p[0] for p in MA_POLYGON), max(p[0] for p in MA_POLYGON)
    lat_min, lat_max = min(p[1] for p in MA_POLYGON), max(p[1] for p in MA_POLYGON)
    lat_idx = np.where((lat >= lat_min - 0.1) & (lat <= lat_max + 0.1))[0]
    lon_idx = np.where((lon >= lon_min - 0.1) & (lon <= lon_max + 0.1))[0]

    lon_grid, lat_grid = np.meshgrid(lon[lon_idx], lat[lat_idx])
    sub_mask = point_in_polygon(lon_grid, lat_grid, MA_POLYGON)

    mask = np.zeros((lat.size, lon.size), dtype=bool)
    mask[np.ix_(lat_idx, lon_idx)] = sub_mask
    return mask


def daily_ma_mean(path, mask):
    """Returns a pandas Series of MA-averaged specific humidity, indexed by date."""
    ds = nc.Dataset(path)
    day_var = ds.variables["day"]
    dates = nc.num2date(day_var[:], units=day_var.units, calendar=day_var.calendar)
    dates = pd.to_datetime([d.isoformat() for d in dates]).normalize()

    sph = ds.variables["specific_humidity"][:]  # (day, lat, lon), masked array
    means = sph[:, mask].mean(axis=1)
    means = np.ma.filled(means, np.nan)
    return pd.Series(means, index=dates, name="specific_humidity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--out", default=str(HERE / "ma_specific_humidity.csv"))
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    series_parts = []
    for fname in ("sph_2025.nc", "sph_2026.nc"):
        ds = nc.Dataset(HERE / fname)
        mask = ma_grid_mask(ds)
        series_parts.append(daily_ma_mean(HERE / fname, mask))

    combined = pd.concat(series_parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    full_range = pd.date_range(start, end, freq="D")
    result = combined.reindex(full_range)
    result.index.name = "date"

    missing = result[result.isna()].index
    if len(missing):
        print(
            f"Warning: {len(missing)} day(s) in the requested range have no data "
            f"(likely not yet published) — from {missing.min().date()} to {missing.max().date()}."
        )

    result.to_csv(args.out, header=True, date_format="%Y-%m-%d")
    print(f"Wrote {len(result)} rows to {args.out}")


if __name__ == "__main__":
    main()
