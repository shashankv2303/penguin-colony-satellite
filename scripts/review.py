"""Review downloaded tiles by eye and relabel them by what they actually show.

Step 1  python3 scripts/review.py sheets
        -> dataset/review/sheet_01.png, sheet_02.png ... (20 numbered thumbnails per sheet)
        -> dataset/review/labels.csv with one row per tile (new_group left blank)

Step 2  open the sheets, fill the new_group column in labels.csv with one of
            colony            reddish-brown stain visible (compare with Cape Gates IMG_007)
            confuser          could be mistaken for guano: rock, edge fringes, shadow, cloud
            clean_ice_or_sea  plain ice, snow or water
            uncertain         might be a stain but cannot be confirmed at 10 m
            reject           blank, broken or a near-duplicate
        leave it blank to keep the current group.

Step 3  python3 scripts/review.py apply
        -> moves files between dataset/raw/<group>/ folders and updates metadata.csv
        -> rejected tiles go to dataset/rejected/ (nothing is deleted)
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"
META = DS / "metadata.csv"
REVIEW = DS / "review"
LABELS = REVIEW / "labels.csv"
REJ_DIR = DS / "rejected"
REJECTED = REJ_DIR / "rejected.csv"
GROUPS = {"colony", "confuser", "clean_ice_or_sea", "uncertain"}
THUMB, COLS, PER_SHEET = 220, 5, 20


def sheets():
    meta = pd.read_csv(META)
    REVIEW.mkdir(exist_ok=True)
    for old in REVIEW.glob("sheet_*.png"):
        old.unlink()
    for s in range(0, len(meta), PER_SHEET):
        tiles = []
        for _, row in meta.iloc[s:s + PER_SHEET].iterrows():
            img = cv2.resize(cv2.imread(str(DS / "raw" / row.filename)), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            label = f"{Path(row.filename).stem} {row.location.split(' (')[0][:16]}"
            img = cv2.copyMakeBorder(img, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % COLS:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
        out = REVIEW / f"sheet_{s // PER_SHEET + 1:02d}.png"
        cv2.imwrite(str(out), grid)
        print("wrote", out.relative_to(ROOT))
    def nodata_pct(f):  # pure black in all channels = outside the satellite swath
        a = cv2.imread(str(DS / "raw" / f))
        return round(100 * (a.max(axis=2) == 0).mean(), 1)
    labels = meta[["filename", "group"]].assign(nodata_pct=meta.filename.map(nodata_pct),
                                                location=meta.location, date=meta.date, new_group="")
    if LABELS.exists():  # keep anything already filled in
        prev = pd.read_csv(LABELS, dtype=str).fillna("")
        labels = labels.drop(columns="new_group").merge(prev[["filename", "new_group"]], on="filename", how="left").fillna("")
    labels.to_csv(LABELS, index=False)
    print("fill new_group in", LABELS.relative_to(ROOT))


def apply():
    meta = pd.read_csv(META)
    labels = pd.read_csv(LABELS, dtype=str).fillna("")
    bad = set(labels.new_group) - GROUPS - {"", "reject"}
    if bad:
        sys.exit(f"Unknown labels {bad}; use {sorted(GROUPS)} or reject")
    changes = labels[(labels.new_group != "") & (labels.new_group != labels.group)]
    rejected_rows = []
    for _, lab in changes.iterrows():
        i = meta.index[meta.filename == lab.filename][0]
        src = DS / "raw" / lab.filename
        name = Path(lab.filename).name
        if lab.new_group == "reject":
            REJ_DIR.mkdir(exist_ok=True)
            shutil.move(src, REJ_DIR / name)
            rejected_rows.append(meta.loc[i].to_dict())
            meta = meta.drop(i)
            print(f"rejected {lab.filename}")
        else:
            dst = DS / "raw" / lab.new_group / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            meta.loc[i, ["filename", "group"]] = [f"{lab.new_group}/{name}", lab.new_group]
            print(f"{lab.filename} -> {lab.new_group}")
    meta.to_csv(META, index=False)
    if rejected_rows:
        old = pd.read_csv(REJECTED) if REJECTED.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rejected_rows)], ignore_index=True).to_csv(REJECTED, index=False)
    LABELS.unlink()  # labels refer to old paths now; regenerate with `sheets`
    print(f"{len(changes)} changes applied; metadata.csv has {len(meta)} rows:", meta.group.value_counts().to_dict())


def merge():
    """python3 scripts/review.py merge path/to/returned_labels.csv  (fills new_group from someone else's file)"""
    if len(sys.argv) < 3:
        sys.exit(merge.__doc__)
    mine = pd.read_csv(LABELS, dtype=str).fillna("")
    theirs = pd.read_csv(sys.argv[2], dtype=str).fillna("")
    theirs["new_group"] = theirs.new_group.str.strip().str.lower().replace({"clean": "clean_ice_or_sea"})
    bad = set(theirs.new_group) - GROUPS - {"", "reject"}
    if bad:
        sys.exit(f"Unknown labels {bad}; use {sorted(GROUPS)} or reject")
    got = theirs[theirs.new_group != ""].set_index("filename").new_group
    unknown = set(got.index) - set(mine.filename)
    if unknown:
        sys.exit(f"Files not in labels.csv: {sorted(unknown)[:5]}")
    mine.loc[mine.filename.isin(got.index), "new_group"] = mine.filename.map(got)
    mine.to_csv(LABELS, index=False)
    print(f"merged {len(got)} labels; still blank: {(mine.new_group == '').sum()}")


if __name__ == "__main__":
    {"sheets": sheets, "apply": apply, "merge": merge}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
