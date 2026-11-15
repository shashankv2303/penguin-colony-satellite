"""Rebuild handoff/review_package.zip (tiles + contact sheets + labels.csv + PDF guide) for a teammate to label.

Usage:  python3 scripts/build_package.py
Run it after downloading, so the package contains every tile currently in dataset/raw/.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS, OUT = ROOT / "dataset", ROOT / "handoff"
PKG = OUT / "review_package"

subprocess.run([sys.executable, str(ROOT / "scripts" / "review.py"), "sheets"], check=True, stdout=subprocess.DEVNULL)

guide = OUT / "Labelling_Guide.pdf"
if not guide.exists():
    sys.exit(f"Missing {guide}; keep the PDF guide in handoff/ first.")

shutil.rmtree(PKG, ignore_errors=True)
(PKG / "sheets").mkdir(parents=True)
(PKG / "tiles").mkdir()
for f in sorted((DS / "review").glob("sheet_*.png")):
    shutil.copy(f, PKG / "sheets" / f.name)
shutil.copy(DS / "review" / "labels.csv", PKG / "labels.csv")
shutil.copy(guide, PKG / guide.name)
n = 0
for f in sorted((DS / "raw").rglob("IMG_*.png")):
    shutil.copy(f, PKG / "tiles" / f.name)
    n += 1
zip_path = shutil.make_archive(str(OUT / "review_package"), "zip", OUT, "review_package")
print(f"{n} tiles packaged -> {Path(zip_path).relative_to(ROOT)} ({Path(zip_path).stat().st_size / 1e6:.0f} MB)")
