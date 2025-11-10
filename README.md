SATSAR Extract
==============

Chips runway polygons from a GeoTIFF into GeoTIFF tiles with preserved CRS.

Install
-------
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Usage
-----
python extract_satsar.py <IMAGEFILENAME> <DATAFILENAME> [--outdir OUTDIR] [--jpg-preview]

To batch run:
Get-ChildItem -Path data -Filter "*_Airport.tif" | ForEach-Object {
    python satsar_extract.py $_.Name data/Airports.fixed.csv
}


Inputs
------
- GeoTIFF (any CRS).
- CSV/XLSX with WGS-84 coordinates and names:
  airfield_name, airstrip_name, lon1,lat1, lon2,lat2, lon3,lat3, lon4,lat4
  (x1/y1..x4/y4 also supported, x = lon, y = lat)

Output
------
- GeoTIFF chips to ./chips/<image_stem>/  (lossless LZW compression)
- Optional JPG preview when --jpg-preview is set.








==============
Helper Script
fix_runway_corners.py

- Some runway coordinates may produce self-crossing (“bow-tie”) shapes.
- Run this helper first to automatically reorder corner points and create a corrected CSV:
- python fix_runway_corners.py data/Airports.csv --out data/Airports.fixed.csv

- It writes a new file (e.g. Airports.fixed.csv) with valid, properly ordered corners.
- Use that file as input to extract_satsar.py for clean, rectangular runway chips.