"""Preprocessing for the penguin-colony satellite dataset.

    python3 scripts/preprocess.py

Writes (raw tiles are never modified):
    dataset/processed/<technique>/IMG_###.png   each technique on its own, for a DEMO set of tiles
    dataset/processed/final/<group>/IMG_###.png  the combined pipeline, for every tile in dataset/raw
    dataset/processed/normalized/<group>/IMG_###.npy  float32 [0,1] version of the final tiles
    dataset/processed/normalization_stats.json   per-channel mean/std of the final tiles

Why these techniques is explained in report/preprocessing_report (see the document).
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"
OUT = DS / "processed"

# One tile per situation, so each technique can be shown on the case it is meant for.
DEMO = ["IMG_007", "IMG_008", "IMG_010", "IMG_011", "IMG_055", "IMG_056", "IMG_068",   # stains
        "IMG_004", "IMG_001",                                                           # confusers (cliff edge, rock/glacier)
        "IMG_081", "IMG_092",                                                           # plain ice off-site
        "IMG_045"]                                                                      # black no-data wedge
TARGET = 256        # final tile size in pixels
GAMMA = 1.8
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)


# ---------- helpers ----------
def find_tile(stem):
    for folder in ("raw", "rejected"):
        hits = list((DS / folder).rglob(f"{stem}.png"))
        if hits:
            return hits[0]
    raise FileNotFoundError(stem)


def valid_mask(bgr):
    """False where the satellite had no data (pure black in every channel)."""
    return bgr.max(axis=2) > 0


def largest_rectangle(mask):
    """Largest axis-aligned rectangle containing only True pixels -> (y0, y1, x0, x1)."""
    h, w = mask.shape
    heights = np.zeros(w, int)
    best, box = 0, (0, h, 0, w)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []  # (start_x, height)
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                if sh * (x - sx) > best:
                    best, box = sh * (x - sx), (y - sh + 1, y + 1, sx, x)
                start = sx
            stack.append((start, cur))
    return box


# ---------- the techniques (each takes and returns a uint8 BGR image) ----------
def crop_valid(bgr):
    """Cropping: remove the black no-data wedge by keeping the largest fully valid rectangle."""
    m = valid_mask(bgr)
    if m.all():
        return bgr
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    y0, y1, x0, x1 = largest_rectangle(m)
    return bgr[y0:y1, x0:x1]


def resize(bgr, size=TARGET):
    """Resizing: one fixed size; INTER_AREA is the right choice when shrinking."""
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def white_balance(bgr):
    """Colour correction: ice and snow should be neutral white. We average the colour of the bright, unsaturated
    pixels (the ice itself) and scale each channel so that average becomes grey. The very brightest pixels are
    ignored because they are clipped white in every channel and would hide the colour cast.
    Gains are limited so a mostly dark scene (open water) cannot be pushed to extreme colours."""
    m = valid_mask(bgr)
    lum = bgr.astype(np.float32).mean(axis=2)
    bright = m & (lum > np.percentile(lum[m], 60) if m.any() else m) & (bgr.max(axis=2) < 250)
    if bright.sum() < 500:                      # too little ice to judge: leave the image alone
        return bgr
    mean = bgr[bright].astype(np.float32).mean(axis=0)
    gains = np.clip(mean.mean() / np.maximum(mean, 1), 0.8, 1.3)
    out = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    out[~m] = 0
    return out


def contrast_stretch(bgr):
    """Contrast stretching: ONE gain/offset for all channels, taken from brightness. (Stretching each channel
    separately turns neutral shadows reddish, which would fake a stain.)"""
    m = valid_mask(bgr)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g[m], [1, 99.5]) if m.any() else (0, 255)
    out = np.clip((bgr.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    out[~m] = 0
    return out


def gamma_correct(bgr, gamma=GAMMA):
    """Intensity transformation: gamma > 1 spreads out the bright end where ice and a faint stain compete."""
    lut = (255 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def hist_equalize(bgr):
    """Global histogram equalisation of brightness. Shown to explain why CLAHE is preferred."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe(bgr):
    """Contrast enhancement: adaptive (local) histogram equalisation on brightness only, so colours are untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(bgr):
    """Noise removal: 3x3 median filter keeps edges (ice cliffs, iceberg outlines) sharp, unlike Gaussian blur."""
    return cv2.medianBlur(bgr, 3)


def to_gray(bgr):
    """Grayscale conversion: shown to prove the stain is a COLOUR signal that brightness alone loses."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def stain_map(bgr):
    """Colour-space conversion: in Lab, guano is positive in a* (red) and b* (yellow) while ice sits near 0.
    Output is a grey map (brighter = browner). The scale is FIXED, not per image, so clean tiles stay dark."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = (lab[..., 1] - 128) + (lab[..., 2] - 128)
    out = np.clip((s + 5) / 25 * 255, 0, 255).astype(np.uint8)   # ice ~ 0 -> dark grey, stain ~ +9 or more -> bright
    out[~valid_mask(bgr)] = 0
    return out


def rotate90(bgr):
    """Geometric operation: a stain has no preferred orientation, so rotated copies are equally valid samples."""
    return np.ascontiguousarray(np.rot90(bgr))


def final_pipeline(bgr):
    """crop no-data -> colour balance -> CLAHE -> resize."""
    return resize(clahe(white_balance(crop_valid(bgr))))


TECHNIQUES = {
    "01_crop_valid": crop_valid,
    "02_resize": resize,
    "03_white_balance": white_balance,
    "04_contrast_stretch": lambda b: contrast_stretch(white_balance(b)),
    "05_gamma": lambda b: gamma_correct(white_balance(b)),
    "06_hist_equalize": hist_equalize,
    "07_clahe": lambda b: clahe(white_balance(b)),
    "08_denoise_median": lambda b: denoise(white_balance(b)),
    "09_grayscale": lambda b: to_gray(white_balance(b)),
    "10_lab_stain_map": lambda b: stain_map(white_balance(b)),
    "11_rotate90": rotate90,
}


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    for stem in DEMO:
        bgr = cv2.imread(str(find_tile(stem)))
        for name, fn in TECHNIQUES.items():
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stem}.png"), fn(bgr))
    print(f"techniques: {len(TECHNIQUES)} x {len(DEMO)} demo tiles")

    sums, sqs, n = np.zeros(3), np.zeros(3), 0
    tiles = sorted((DS / "raw").rglob("IMG_*.png"))
    for f in tiles:
        group = f.parent.name
        out = final_pipeline(cv2.imread(str(f)))
        (OUT / "final" / group).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / "final" / group / f.name), out)
        x = out[..., ::-1].astype(np.float32) / 255.0          # RGB in [0,1]
        (OUT / "normalized" / group).mkdir(parents=True, exist_ok=True)
        np.save(OUT / "normalized" / group / f"{f.stem}.npy", x)
        sums += x.reshape(-1, 3).sum(0); sqs += (x.reshape(-1, 3) ** 2).sum(0); n += x.shape[0] * x.shape[1]
    mean = sums / n
    std = np.sqrt(sqs / n - mean ** 2)
    (OUT / "normalization_stats.json").write_text(json.dumps(
        {"channels": "RGB", "range": "[0,1]", "mean": mean.round(4).tolist(), "std": std.round(4).tolist(), "tiles": len(tiles)}, indent=2))
    print(f"final pipeline: {len(tiles)} tiles | mean {mean.round(3).tolist()} std {std.round(3).tolist()}")


if __name__ == "__main__":
    main()
