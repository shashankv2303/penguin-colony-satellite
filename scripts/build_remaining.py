"""Package ONLY the still-unlabelled tiles for a teammate: handoff/review_remaining.zip

Usage:  python3 scripts/build_remaining.py
Teammate fills new_group in labels_remaining.csv and sends it back; you then run
    python3 scripts/review.py merge path/to/labels_remaining.csv
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS, OUT = ROOT / "dataset", ROOT / "handoff"
PKG = OUT / "review_remaining"
THUMB, COLS, PER = 220, 5, 20

labels = pd.read_csv(DS / "review" / "labels.csv", dtype=str).fillna("")
todo = labels[labels.new_group == ""].reset_index(drop=True)
if todo.empty:
    raise SystemExit("Nothing left to label.")

shutil.rmtree(PKG, ignore_errors=True)
(PKG / "sheets").mkdir(parents=True)
(PKG / "tiles").mkdir()
for f in todo.filename:
    shutil.copy(DS / "raw" / f, PKG / "tiles" / Path(f).name)

for s in range(0, len(todo), PER):
    tiles = []
    for _, row in todo.iloc[s:s + PER].iterrows():
        img = cv2.resize(cv2.imread(str(DS / "raw" / row.filename)), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        label = f"{Path(row.filename).stem} {row.location.split(' (')[0][:16]}"
        img = cv2.copyMakeBorder(img, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % COLS:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    cv2.imwrite(str(PKG / "sheets" / f"sheet_{s // PER + 1:02d}.png"), grid)

todo.to_csv(PKG / "labels_remaining.csv", index=False)
shutil.copy(OUT / "Labelling_Guide.pdf", PKG / "Labelling_Guide.pdf")
zip_path = shutil.make_archive(str(OUT / "review_remaining"), "zip", OUT, "review_remaining")
print(f"{len(todo)} tiles -> {Path(zip_path).relative_to(ROOT)} ({Path(zip_path).stat().st_size / 1e6:.0f} MB)")
