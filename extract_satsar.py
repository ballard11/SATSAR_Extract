#!/usr/bin/env python3
"""
satsar_extract.py

Chip runways from a GeoTIFF given 4-corner coordinates per runway.

Usage:
  python satsar_extract.py <IMAGEFILENAME> <DATAFILENAME> [--outdir OUTDIR]

Notes:
- Expects runway coordinates in WGS-84 (EPSG:4326). (Matches Google Earth / team standard.)
- Will reproject polygons to the raster's native CRS before masking.
- Supports CSV or Excel (.xlsx/.xls).
- Accepts column schemes:
    (A) lon1/lat1 ... lon4/lat4
    (B) x1/y1 ... x4/y4   (x=longitude, y=latitude)
- Outputs GeoTIFF chips to ./chips/<image_stem>/ by default, or to --outdir if provided.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import geopandas as gpd


# ----------------------------
# Utilities
# ----------------------------

def load_table(path: Path) -> pd.DataFrame:
    lower = path.suffix.lower()
    if lower == ".csv":
        return pd.read_csv(path)
    if lower in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file type for {path}. Use .csv or .xlsx/.xls")


def detect_column_mapping(columns) -> dict:
    """Return a mapping {'lon1': '...', 'lat1': '...', ...} using either lon/lat or x/y scheme."""
    cols = {c.lower(): c for c in columns}  # case-insensitive lookup -> original name
    # Scheme A: lon/lat
    needed_a = ["lon1", "lat1", "lon2", "lat2", "lon3", "lat3", "lon4", "lat4"]
    if all(c in cols for c in needed_a):
        return {k: cols[k] for k in needed_a}

    # Scheme B: x/y (x=lon, y=lat)
    needed_b = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    if all(c in cols for c in needed_b):
        return {
            "lon1": cols["x1"], "lat1": cols["y1"],
            "lon2": cols["x2"], "lat2": cols["y2"],
            "lon3": cols["x3"], "lat3": cols["y3"],
            "lon4": cols["x4"], "lat4": cols["y4"],
        }

    raise ValueError(
        "Could not find required coordinate columns. "
        "Expected either lon1/lat1..lon4/lat4 OR x1/y1..x4/y4."
    )


def safe_stem(s: str) -> str:
    """File-system safe name (keep it simple)."""
    s = (s or "").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def polygon_from_row(row, colmap) -> Polygon:
    pts = [
        (row[colmap["lon1"]], row[colmap["lat1"]]),
        (row[colmap["lon2"]], row[colmap["lat2"]]),
        (row[colmap["lon3"]], row[colmap["lat3"]]),
        (row[colmap["lon4"]], row[colmap["lat4"]]),
    ]
    return Polygon(pts)


def to_raster_crs(poly_wgs84: Polygon, raster_crs) -> Polygon:
    gdf = gpd.GeoDataFrame({"geometry": [poly_wgs84]}, crs="EPSG:4326")
    gdf = gdf.to_crs(raster_crs)
    return gdf.geometry.iloc[0]


# ----------------------------
# Main worker
# ----------------------------

def run_extract(image_filename: Path, data_filename: Path, output_dir: Path) -> list[Path]:
    df = load_table(data_filename)

    # flexible column names for airfield/airstrip
    name_cols = {c.lower(): c for c in df.columns}
    airfield_col = name_cols.get("airfield_name") or name_cols.get("airfield") or name_cols.get("airport") or None
    airstrip_col = name_cols.get("airstrip_name") or name_cols.get("airstrip") or name_cols.get("runway") or None
    if airfield_col is None or airstrip_col is None:
        raise ValueError("Missing airfield/airstrip name columns (expected 'airfield_name' and 'airstrip_name').")

    colmap = detect_column_mapping(df.columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    chips_written: list[Path] = []

    with rasterio.open(image_filename) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {image_filename}")

        raster_crs = src.crs
        # Use stem so output prefix doesn't include ".tif"
        base_tif = Path(image_filename).stem

        for idx, row in df.iterrows():
            airfield = str(row[airfield_col]).strip()
            airstrip = str(row[airstrip_col]).strip()

            # polygon in WGS84 (team standard)
            poly_wgs84 = polygon_from_row(row, colmap)

            # same polygon projected into raster CRS
            try:
                poly_raster = to_raster_crs(poly_wgs84, raster_crs)
            except Exception as e:
                print(f"[WARN] Row {idx} ({airfield}/{airstrip}): reprojection failed: {e}")
                continue

            # --- CROP ---
            try:
                chip_arr, out_transform = mask(src, [poly_raster], crop=True)
            except ValueError as e:
                print(f"[WARN] Row {idx} ({airfield}/{airstrip}): no overlap: {e}")
                continue
            except Exception as e:
                print(f"[WARN] Row {idx} ({airfield}/{airstrip}): mask failed: {e}")
                continue

            # --- SAVE AS GEOTIFF ---
            safe_airfield = safe_stem(airfield)
            safe_airstrip = safe_stem(airstrip)
            out_name = f"{base_tif}.{safe_airfield}.{safe_airstrip}.tif"
            out_path = output_dir / out_name

            try:
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": chip_arr.shape[1],
                    "width":  chip_arr.shape[2],
                    "transform": out_transform,
                    "compress": "LZW",
                    "count": src.count,           # <— be explicit
                    "dtype": src.dtypes[0],       # lossless
                })

                # Ensure contiguous memory for safety
                chip_to_write = np.ascontiguousarray(chip_arr)

                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(chip_to_write)

                print(f"[OK] {out_path}")
                chips_written.append(out_path)
            except Exception as e:
                print(f"[WARN] Row {idx} ({airfield}/{airstrip}): save failed: {e}")

    return chips_written


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Chip runways from a GeoTIFF using 4-corner coordinates.")
    parser.add_argument("image_filename", help="Path to GeoTIFF image")
    parser.add_argument("data_filename", help="Path to CSV/XLSX with runway corners and names")
    parser.add_argument(
        "--outdir",
        help="Output directory for chips (default: ./chips/<image_stem>/)",
        default=None,
    )
    args = parser.parse_args()

    image_path = Path(args.image_filename)
    data_path = Path(args.data_filename)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Default output folder is relative, organized by image stem
    out_dir = Path(args.outdir) if args.outdir else Path("chips") / image_path.stem

    chips = run_extract(image_path, data_path, out_dir)
    print(f"[DONE] Wrote {len(chips)} chips to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
