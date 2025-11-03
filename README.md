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
