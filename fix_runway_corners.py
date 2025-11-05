#!/usr/bin/env python3
"""
fix_runway_corners.py

Auto-fix misordered runway corner coordinates (bow-tie/self-crossing quads).
Reads a CSV/XLSX with 4-corner columns and writes a NEW CSV with corrected order.

Usage:
  python fix_runway_corners.py <INPUT_TABLE> [--out OUTPUT_CSV]

Notes:
- Supports CSV and Excel (.xlsx/.xls).
- Accepts column schemes:
    (A) lon1/lat1 ... lon4/lat4
    (B) x1/y1 ... x4/y4   (x=longitude, y=latitude)
- Adds a boolean column `_fixed` indicating rows that were changed.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from math import atan2
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPoint

# ----------------------------
# Column detection (same idea as your main script)
# ----------------------------

def load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table type: {path.suffix}. Use .csv or .xlsx/.xls")

def detect_column_mapping(columns) -> dict:
    cols = {c.lower(): c for c in columns}  # case-insensitive to original
    need_a = ["lon1","lat1","lon2","lat2","lon3","lat3","lon4","lat4"]
    if all(k in cols for k in need_a):
        return {k: cols[k] for k in need_a}

    need_b = ["x1","y1","x2","y2","x3","y3","x4","y4"]
    if all(k in cols for k in need_b):
        return {
            "lon1": cols["x1"], "lat1": cols["y1"],
            "lon2": cols["x2"], "lat2": cols["y2"],
            "lon3": cols["x3"], "lat3": cols["y3"],
            "lon4": cols["x4"], "lat4": cols["y4"],
        }

    raise ValueError(
        "Could not find 4-corner columns. "
        "Expected lon1/lat1..lon4/lat4 OR x1/y1..x4/y4."
    )

# ----------------------------
# Geometry helpers
# ----------------------------

def polygon_from_points(pts):
    """pts: list of (lon, lat) length 4"""
    try:
        return Polygon(pts)
    except Exception:
        return Polygon()

def is_valid_quad(pts) -> bool:
    poly = polygon_from_points(pts)
    return (poly.is_valid and not poly.is_empty and poly.area > 0)

def angle_sort_ccw(pts):
    """Return points sorted CCW around centroid."""
    arr = np.asarray(pts, dtype=float)
    cx, cy = arr.mean(axis=0)
    angles = [atan2(y - cy, x - cx) for x, y in arr]
    order = np.argsort(angles)  # CCW
    return [pts[i] for i in order]

def rotate_start_to_closest(sorted_pts, ref_pt):
    """Rotate list so it starts from the element closest to ref_pt."""
    dists = [ (i, (p[0]-ref_pt[0])**2 + (p[1]-ref_pt[1])**2) for i,p in enumerate(sorted_pts) ]
    start = min(dists, key=lambda t: t[1])[0]
    return sorted_pts[start:] + sorted_pts[:start]

def convex_hull_order(pts):
    """Return hull vertices in order if hull has 4 unique points, else None."""
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    if not hull or hull.is_empty:
        return None
    # Hull for 4 points should be a Polygon with 5 coords (closed ring)
    if isinstance(hull, Polygon):
        coords = list(hull.exterior.coords)[:-1]
        if len(coords) == 4:
            return [(x, y) for x, y in coords]
    return None

def fix_quad_order(pts):
    """
    Given 4 points [(x,y)*4], return:
      fixed_pts (list), changed (bool), method (str)
    """
    # Already good?
    if is_valid_quad(pts):
        return pts, False, "original"

    # 1) Try CCW by angle around centroid
    ccw = angle_sort_ccw(pts)
    # start from the closest to original first point to keep 'lon1/lat1' near original
    ccw_rot = rotate_start_to_closest(ccw, pts[0])
    if is_valid_quad(ccw_rot):
        return ccw_rot, True, "ccw_centroid_sort"

    # 2) Try convex hull order (ensures non-self-crossing if 4 unique corners)
    hull = convex_hull_order(pts)
    if hull is not None:
        hull_rot = rotate_start_to_closest(hull, pts[0])
        if is_valid_quad(hull_rot):
            return hull_rot, True, "convex_hull"

    # 3) Give up—return original but mark as unchanged (still invalid)
    return pts, False, "unchanged_invalid"

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Auto-fix misordered runway corners and write a NEW CSV.")
    ap.add_argument("input_table", help="Path to CSV/XLSX with 4-corner columns")
    ap.add_argument("--out", help="Output CSV path (default: <stem>.fixed.csv beside input)", default=None)
    args = ap.parse_args()

    in_path = Path(args.input_table)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + ".fixed.csv")

    df = load_table(in_path)
    colmap = detect_column_mapping(df.columns)

    # prepare output copy
    out_df = df.copy()
    fixed_flags = []
    methods = []

    for idx, row in df.iterrows():
        pts = [
            (row[colmap["lon1"]], row[colmap["lat1"]]),
            (row[colmap["lon2"]], row[colmap["lat2"]]),
            (row[colmap["lon3"]], row[colmap["lat3"]]),
            (row[colmap["lon4"]], row[colmap["lat4"]]),
        ]

        # handle NaNs gracefully
        if any(pd.isna(x) or pd.isna(y) for x, y in pts):
            fixed_flags.append(False)
            methods.append("skipped_nan")
            continue

        fixed_pts, changed, method = fix_quad_order(pts)
        (lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4) = fixed_pts

        # write back (always write; only 'changed' flags differ)
        out_df.at[idx, colmap["lon1"]] = lon1
        out_df.at[idx, colmap["lat1"]] = lat1
        out_df.at[idx, colmap["lon2"]] = lon2
        out_df.at[idx, colmap["lat2"]] = lat2
        out_df.at[idx, colmap["lon3"]] = lon3
        out_df.at[idx, colmap["lat3"]] = lat3
        out_df.at[idx, colmap["lon4"]] = lon4
        out_df.at[idx, colmap["lat4"]] = lat4

        fixed_flags.append(changed)
        methods.append(method)

    out_df["_fixed"] = fixed_flags
    out_df["_fix_method"] = methods

    out_df.to_csv(out_path, index=False)
    total = len(df)
    changed = sum(1 for f in fixed_flags if f)
    print(f"[DONE] Wrote: {out_path}")
    print(f"[STATS] Rows: {total} | Changed: {changed} | Unchanged: {total - changed}")

if __name__ == "__main__":
    main()
