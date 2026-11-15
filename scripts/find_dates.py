"""Find real Sentinel-2 acquisition dates over a site and (optionally) add them to tile_plan.csv.

Usage examples:
    python3 scripts/find_dates.py --site "Cuverville Island"                  # just list scenes (a diagnosis)
    python3 scripts/find_dates.py --site "Cuverville Island" --append --per-site 4
    python3 scripts/find_dates.py --all-peninsula --append --per-site 4 --max-cloud 40

Needs the same CDSE_CLIENT_ID / CDSE_CLIENT_SECRET environment variables as download_tiles.py.

The cloud value is for the whole ~110 km scene, not just your 5 km tile, so treat it as a hint.
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_tiles import DATA_TYPE, ROOT, PLAN, bbox_around, get_token  # noqa: E402

CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
REFS = [ROOT / "colonies_reference.csv", ROOT / "peninsula_colonies_reference.csv"]
# months when each region is lit and (for the peninsula) mostly snow-free enough to show stains
SEASON = {"peninsula": (12, 3), "continent": (10, 12)}
# Fretwell et al. 2021 Table 1: colonies shown visible in Sentinel-2 (in 3-29 images each)
CONFIRMED = ["Yule Bay", "Karelin Bay", "Casey Bay", "Verdi Inlet", "Cape Gates", "Cruzen Island",
             "Ninnis Bank", "Porpoise Bay", "Cape Darlington", "Cape Poinsett", "Pfrogner Point"]


def find_scenes(token, bbox, start, end, max_cloud):
    """Return {date: lowest_cloud_cover} for scenes over bbox between start and end (ISO strings)."""
    body = {"bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "collections": [DATA_TYPE],
            "limit": 100,
            "fields": {"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]}}
    best = {}
    for _ in range(10):  # follow pagination, at most 10 pages of 100
        r = requests.post(CATALOG_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if r.status_code != 200:
            sys.exit(f"Catalog search failed: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        for f in data.get("features", []):
            d = f["properties"]["datetime"][:10]
            c = f["properties"].get("eo:cloud_cover", 100)
            if c < max_cloud:  # filtered here because the server rejected our filter syntax
                best[d] = min(c, best.get(d, 100))
        nxt = data.get("context", {}).get("next")
        if not nxt:
            break
        body["next"] = nxt
    return best


def spaced(dates_cloud, n, min_gap_days=10):
    """Pick up to n dates, lowest cloud first, at least min_gap_days apart (for variation)."""
    chosen = []
    for d, c in sorted(dates_cloud.items(), key=lambda kv: kv[1]):
        dd = date.fromisoformat(d)
        if all(abs((dd - date.fromisoformat(x[0])).days) >= min_gap_days for x in chosen):
            chosen.append((d, c))
        if len(chosen) == n:
            break
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", action="append", default=[])
    ap.add_argument("--all-peninsula", action="store_true")
    ap.add_argument("--confirmed", action="store_true", help="the 11 Sentinel-2-confirmed colonies")
    ap.add_argument("--years", default="2019-2024", help="e.g. 2019-2024")
    ap.add_argument("--max-cloud", type=float, default=60)
    ap.add_argument("--per-site", type=int, default=4)
    ap.add_argument("--append", action="store_true", help="add chosen dates to tile_plan.csv")
    ap.add_argument("--near-colony", action="store_true",
                    help="search +-45 days around dates already labelled 'colony' in dataset/review/labels.csv")
    ap.add_argument("--gap", type=int, default=10, help="minimum days between picked dates")
    args = ap.parse_args()

    refs = pd.concat([pd.read_csv(p).assign(region="peninsula" if "peninsula" in p.name else "continent")
                      for p in REFS if p.exists()], ignore_index=True)
    anchors = {}
    if args.near_colony:  # {site: [dates labelled colony]}: stains persist within a season, so look near those dates
        lab = pd.read_csv(ROOT / "dataset" / "review" / "labels.csv", dtype=str).fillna("")
        lab = lab[lab.new_group.str.strip().str.lower() == "colony"]
        for loc, d in zip(lab.location, lab.date):
            anchors.setdefault(loc.split(" (")[0], []).append(d[:10])
    names = args.site + (list(refs[refs.region == "peninsula"]["name"]) if args.all_peninsula else []) \
        + (CONFIRMED if args.confirmed else []) + list(anchors)
    if not names:
        sys.exit("Give --site NAME (repeatable), --confirmed, --near-colony or --all-peninsula.")
    y0, y1 = map(int, args.years.split("-"))
    token = get_token()
    plan = pd.read_csv(PLAN)
    have = set(zip(plan["name"], plan.date_from))
    new_rows = []

    for name in names:
        hit = refs[refs["name"] == name]
        if hit.empty:
            print(f"unknown site: {name}")
            continue
        s = hit.iloc[0]
        m0, m1 = SEASON[s.region]
        bbox = bbox_around(s.lat, s.lon)
        found = {}
        if name in anchors:
            for a in anchors[name]:
                d0 = date.fromisoformat(a)
                found.update(find_scenes(token, bbox, (d0 - timedelta(days=45)).isoformat(),
                                         (d0 + timedelta(days=45)).isoformat(), args.max_cloud))
        else:
            for y in range(y0, y1 + 1):
                start, end = (f"{y}-{m0:02d}-01", f"{y + 1}-{m1:02d}-28") if m0 > m1 else (f"{y}-{m0:02d}-01", f"{y}-{m1:02d}-28")
                found.update(find_scenes(token, bbox, start, end, args.max_cloud))
        found = {d: c for d, c in found.items() if (name, d) not in have}  # never re-pick a date already in the plan
        picks = spaced(found, args.per_site, args.gap)
        print(f"{name}: {len(found)} scenes under {args.max_cloud:.0f}% cloud; "
              f"best {min(found.values()) if found else '-'}%; picked {[f'{d} ({c:.0f}%)' for d, c in picks]}")
        for d, c in picks:
            if (name, d) not in have:
                new_rows.append(dict(group="colony", name=name, lat=s.lat, lon=s.lon, date_from=d, date_to=d,
                                     dx_km=0, dy_km=0, notes="; ".join(x for x in [f"scene cloud {c:.0f}%", str(s.get("species_and_size", ""))]
                                                    if x and x != "nan").replace(",", ";")))

    if args.append and new_rows:
        pd.concat([plan, pd.DataFrame(new_rows)], ignore_index=True).to_csv(PLAN, index=False)
        print(f"added {len(new_rows)} rows to tile_plan.csv")
    elif new_rows:
        print("(not saved; add --append to write these into tile_plan.csv)")


if __name__ == "__main__":
    main()
