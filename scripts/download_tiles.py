"""Download Sentinel-2 true-colour tiles listed in tile_plan.csv and log them in dataset/metadata.csv.

Usage:
    python3 scripts/download_tiles.py --dry-run     # no network, just prints what it would request
    python3 scripts/download_tiles.py               # real download (needs credentials, see below)

Credentials (never put these in a file or in chat):
    export CDSE_CLIENT_ID="..."
    export CDSE_CLIENT_SECRET="..."
Create them in the Copernicus Data Space dashboard under "OAuth clients".

tile_plan.csv columns:
    group      colony | clean_ice_or_sea | confuser
    name       label for the location (e.g. a colony name)
    lat, lon   centre of the tile in degrees (south is negative)
    date_from, date_to   search window, YYYY-MM-DD
    dx_km, dy_km         optional offset east / north from lat,lon (for clean or confuser tiles near a colony)
    notes      anything you want recorded
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "tile_plan.csv"
RAW = ROOT / "dataset" / "raw"
META = ROOT / "dataset" / "metadata.csv"
REJECTED = ROOT / "dataset" / "rejected" / "rejected.csv"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# --- Decisions that are yours to understand and change -------------------------------------
TILE_PX = 512       # output size in pixels
M_PER_PX = 10       # Sentinel-2 visible bands are 10 m per pixel, so the tile covers 5.12 km
DATA_TYPE = "sentinel-2-l1c"  # L1C (top-of-atmosphere) has scenes back to 2015; L2A is patchy in Antarctica before 2018
MAX_CLOUD = 40      # percent; scene-level filter, treat it as a hint only (bright ice confuses it)
GAIN = 1.0          # brightness multiplier in the evalscript; 2.5 clipped 86-100% of ice pixels to white
GROUPS = {"colony", "clean_ice_or_sea", "confuser"}
# ---------------------------------------------------------------------------------------------

EVALSCRIPT = f"""//VERSION=3
function setup() {{ return {{ input: ["B02", "B03", "B04"], output: {{ bands: 3 }} }}; }}
function evaluatePixel(s) {{ return [{GAIN}*s.B04, {GAIN}*s.B03, {GAIN}*s.B02]; }}
"""


def bbox_around(lat, lon, dx_km=0.0, dy_km=0.0):
    """Return [west, south, east, north] in degrees for a square tile.

    One degree of latitude is ~111.32 km everywhere, but one degree of longitude shrinks with
    cos(latitude). Near Antarctica cos(lat) is small, so the longitude span must be wider.
    """
    half_km = TILE_PX * M_PER_PX / 1000 / 2
    lat += dy_km / 111.32
    lon += dx_km / (111.32 * math.cos(math.radians(lat)))
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))
    return [round(lon - dlon, 6), round(lat - dlat, 6), round(lon + dlon, 6), round(lat + dlat, 6)]


def get_token():
    cid, secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET first (see the top of this file).")
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                       "client_id": cid, "client_secret": secret}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def build_request(bbox, date_from, date_to):
    return {
        "input": {
            "bounds": {"properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}, "bbox": bbox},
            "data": [{
                "type": DATA_TYPE,
                "dataFilter": {
                    "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                    "maxCloudCoverage": MAX_CLOUD,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {"width": TILE_PX, "height": TILE_PX,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": EVALSCRIPT,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = pd.read_csv(PLAN).fillna({"dx_km": 0, "dy_km": 0, "notes": ""})
    bad = set(plan["group"]) - GROUPS
    if bad:
        sys.exit(f"Unknown group(s) in tile_plan.csv: {bad}. Allowed: {GROUPS}")

    meta = pd.read_csv(META) if META.exists() and META.stat().st_size > 0 else pd.DataFrame()
    rejected = pd.read_csv(REJECTED) if REJECTED.exists() else pd.DataFrame()
    done = set(meta.get("source_url", [])) | set(rejected.get("source_url", []))
    # number after the highest IMG_### ever used, so moved or rejected files never get a reused name
    used = [int(m) for m in pd.concat([meta.get("filename", pd.Series(dtype=str)),
                                        rejected.get("filename", pd.Series(dtype=str))])
            .str.extract(r"IMG_(\d+)")[0].dropna()]
    next_id = max(used, default=0) + 1
    token = None if args.dry_run else get_token()
    new_rows = []

    for _, row in plan.iterrows():
        bbox = bbox_around(row.lat, row.lon, row.dx_km, row.dy_km)
        key = f"{PROCESS_URL}?bbox={bbox}&from={row.date_from}&to={row.date_to}"
        if key in done:
            print(f"skip (already logged): {row['name']} {row.date_from}")
            continue
        fname = f"IMG_{next_id:03d}.png"
        out = RAW / row["group"] / fname
        if args.dry_run:
            print(f"[dry-run] {row['group']:17s} {fname} {row['name']} bbox={bbox} {row.date_from}..{row.date_to}")
            next_id += 1
            continue

        r = requests.post(PROCESS_URL, json=build_request(bbox, row.date_from, row.date_to),
                          headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if r.status_code == 401:  # token expired: get a new one and retry once
            token = get_token()
            r = requests.post(PROCESS_URL, json=build_request(bbox, row.date_from, row.date_to),
                              headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if r.status_code != 200:
            print(f"FAILED {row['name']} {row.date_from}: HTTP {r.status_code} {r.text[:200]}")
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        img = Image.open(out)
        if np.asarray(img).mean() < 1:  # entirely black = no data in that window
            out.unlink()
            print(f"EMPTY (no usable scene) {row['name']} {row.date_from}..{row.date_to}")
            continue

        new_rows.append({
            "filename": f"{row['group']}/{fname}", "group": row["group"],
            "source": f"Copernicus Data Space - {DATA_TYPE} (Sentinel Hub Process API), true colour B04/B03/B02",
            "source_url": key,
            "date": f"{row.date_from} (single acquisition day)" if row.date_from == row.date_to
                    else f"{row.date_from} to {row.date_to} (least-cloud mosaic)",
            "location": f"{row['name']} ({row.lat}, {row.lon}; offset {row.dx_km},{row.dy_km} km)",
            "width": img.width, "height": img.height, "format": img.format, "notes": row["notes"],
        })
        print(f"saved {fname}  {row['group']}  {row['name']}  {row.date_from}")
        next_id += 1

    if new_rows:
        pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True).to_csv(META, index=False)
        print(f"metadata.csv now has {len(meta) + len(new_rows)} rows")


if __name__ == "__main__":
    main()
