"""Before/after figures for the report -> report/figures/<technique>.png  (run after scripts/preprocess.py)."""
import glob
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DS, FIG = ROOT / "dataset", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def raw_of(stem):
    return (glob.glob(str(DS / "raw" / "*" / f"{stem}.png")) + glob.glob(str(DS / "rejected" / f"{stem}.png")))[0]


def load(p, gray=False):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)


def show(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


# technique, tile, title, input is the colour-balanced image?, zoom box (x0, y0, x1, y1)
PAIRS = [("01_crop_valid", "IMG_045", "Cropping", False, None),
         ("02_resize", "IMG_007", "Resizing (512 to 256 px)", False, None),
         ("03_white_balance", "IMG_007", "Colour balance", False, None),
         ("04_contrast_stretch", "IMG_056", "Contrast stretching", True, None),
         ("05_gamma", "IMG_007", "Gamma correction (1.8)", True, None),
         ("06_hist_equalize", "IMG_007", "Global histogram equalisation", False, None),
         ("07_clahe", "IMG_007", "CLAHE", True, None),
         ("08_denoise_median", "IMG_007", "Median filter 3x3 (4x zoom on stain)", True, (240, 330, 400, 490)),
         ("09_grayscale", "IMG_007", "Grayscale conversion", True, None),
         ("10_lab_stain_map", "IMG_007", "Lab stain map", True, None),
         ("11_rotate90", "IMG_007", "Rotation by 90 degrees", False, None)]

for tech, stem, title, balanced_in, box in PAIRS:
    gray_out = tech in ("09_grayscale", "10_lab_stain_map")
    a = load(f"{DS}/processed/03_white_balance/{stem}.png") if balanced_in else load(raw_of(stem))
    b = load(f"{DS}/processed/{tech}/{stem}.png", gray_out)
    if box:
        x0, y0, x1, y1 = box
        a = cv2.resize(a[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(b[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
    ax[0].imshow(show(a)); ax[0].set_title(f"Input: {stem}" + (" (colour-balanced)" if balanced_in else " (raw)"), fontsize=10)
    ax[1].imshow(show(b), cmap="gray" if gray_out else None); ax[1].set_title(f"Output: {title}", fontsize=10)
    ax[0].set_xlabel(f"{a.shape[1]}x{a.shape[0]} px", fontsize=8); ax[1].set_xlabel(f"{b.shape[1]}x{b.shape[0]} px", fontsize=8)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    plt.tight_layout(); plt.savefig(FIG / f"{tech}.png", dpi=130); plt.close()

# histograms: raw vs after the final pipeline, same tile
a = load(raw_of("IMG_056")); c = load(glob.glob(str(DS / "processed" / "final" / "*" / "IMG_056.png"))[0])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for x, img, t in zip(ax, (a, c), ("Raw IMG_056", "After final pipeline")):
    for i, col in enumerate("bgr"):
        x.hist(img[..., i].ravel(), bins=64, range=(0, 255), color=col, alpha=.5, log=True)
    x.set_title(t, fontsize=9); x.set_xlabel("pixel value (log counts)", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "histograms.png", dpi=130); plt.close()
print("figures:", len(list(FIG.glob("*.png"))))
