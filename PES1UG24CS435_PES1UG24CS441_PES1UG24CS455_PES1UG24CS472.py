#!/usr/bin/env python3
"""
Image Processing - Mini Project 2
Team: PES1UG24CS435, PES1UG24CS441, PES1UG24CS455, PES1UG24CS472

Dataset (from Mini Project 1): Sentinel-2 satellite tiles of Antarctic penguin-colony sites, already preprocessed
(crop no-data, colour balance, CLAHE, resize 256x256). Folder layout used:  <dataset>/processed/final/<group>/IMG_###.png
Groups used: colony, confuser, clean_ice_or_sea  ('uncertain' tiles are left out: their label is not reliable).

Tasks
  1. SIFT features  -> Bag of Visual Words -> SVM
     Harris corner features                -> SVM
  2. ResNet-18 (ImageNet-pretrained, fine-tuned)
     MViT-v2 tiny (ImageNet-pretrained, fine-tuned; timm)

Two settings are evaluated for every method:  'three' (colony / confuser / clean_ice_or_sea)  and  'binary' (colony vs other).

Evaluation (why it looks like this)
  * Tiles of the same colony site on different dates look almost identical, so a random split would leak near-duplicates
    into the test set. We use StratifiedGroupKFold: every site is entirely in train OR test in each fold.
  * Only ~20% of tiles are colonies, so plain accuracy is misleading. We report macro-F1 and the colony class's
    precision / recall / F1, computed on the pooled out-of-fold predictions.

Usage
  python <this_file>.py --data path/to/dataset            # everything
  python <this_file>.py --tasks sift,harris               # classical part only (needs no PyTorch)
  python <this_file>.py --tasks resnet,mvit --epochs 15   # deep learning only (GPU recommended: Colab / Kaggle)

Requirements: numpy pandas scikit-learn opencv-python (SIFT in cv2 >= 4.4); for task 2 also: torch torchvision timm pillow
  Colab / Kaggle:  !pip install -q timm
"""
import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

CLASSES = ["clean_ice_or_sea", "colony", "confuser"]   # alphabetical; label ids 0, 1, 2
COLONY = 1                                             # colony is id 1 in both settings (binary: other=0, colony=1)
SEED = 42


# ----------------------------------------------------------------------------- data
def find_final_dir(user_path):
    """Locate .../processed/final (must contain one folder per class). Searches common Colab / Kaggle places too."""
    roots = ([Path(user_path)] if user_path else []) + [Path.cwd(), Path("/content"), Path("/kaggle/input"),
                                                        Path("/kaggle/working")]
    ok = lambda p: p.is_dir() and all((p / c).is_dir() for c in CLASSES)
    for r in roots:
        if not r.exists():
            continue
        for cand in (r, r / "final", r / "processed" / "final", r / "dataset" / "processed" / "final"):
            if ok(cand):
                return cand
        for cand in r.glob("**/processed/final"):
            if ok(cand):
                return cand
    sys.exit("Could not find processed/final/<group> folders. Pass the dataset folder with --data.")


def load_samples(final_dir):
    rows = [dict(path=str(f), stem=f.stem, y3=i) for i, c in enumerate(CLASSES) for f in sorted((final_dir / c).glob("*.png"))]
    df = pd.DataFrame(rows)
    # site = the colony location the tile belongs to; used as the group for the split
    meta = next((m for m in (final_dir.parent.parent / "metadata.csv", final_dir.parent / "metadata.csv",
                             final_dir / "metadata.csv") if m.exists()), None)
    if meta is None:
        print("WARNING: metadata.csv not found -> cannot group tiles by site; falling back to a stratified split "
              "(results will be optimistic).")
        df["site"] = df.stem
    else:
        m = pd.read_csv(meta)
        m["stem"] = m.filename.str.extract(r"(IMG_\d+)")[0]
        m["site"] = m.location.str.split(" (", regex=False).str[0].str.replace(" offset", "", regex=False)
        df["site"] = df.stem.map(dict(zip(m.stem, m.site))).fillna("unknown_" + df.stem)
    return df


def make_folds(df, n_splits):
    if df.site.nunique() >= n_splits and not df.site.str.startswith("unknown_").all() and df.site.nunique() < len(df):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(df, df.y3, groups=df.site))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(df, df.y3))


def labels_for(df, setting):
    if setting == "binary":
        return (df.y3.values == COLONY).astype(int), ["other", "colony"]
    return df.y3.values.copy(), CLASSES


# ----------------------------------------------------------------------------- metrics / CV driver
def score(y, pred, names):
    labs = list(range(len(names)))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[COLONY], zero_division=0)
    return dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=labs, average="macro", zero_division=0),
                colony_p=p[0], colony_r=r[0], colony_f1=f[0], cm=confusion_matrix(y, pred, labels=labs))


def cross_validate(folds, y, fit_predict):
    """fit_predict(train_idx, test_idx) -> predicted labels for test_idx. Returns pooled out-of-fold predictions."""
    oof, accs = np.full(len(y), -1), []
    for k, (tr, te) in enumerate(folds):
        pred = fit_predict(tr, te)
        oof[te] = pred
        accs.append(accuracy_score(y[te], pred))
    return oof, accs


def report(rows, method, setting, y, oof, accs, names, seconds):
    s = score(y, oof, names)
    print(f"\n[{method} | {setting}]  acc {s['acc']:.3f} | macro-F1 {s['macro_f1']:.3f} | colony P/R/F1 "
          f"{s['colony_p']:.2f}/{s['colony_r']:.2f}/{s['colony_f1']:.2f} | fold acc {np.mean(accs):.3f}+-{np.std(accs):.3f} | {seconds:.0f}s")
    print("confusion matrix (rows = true, cols = predicted), order:", names)
    print(s["cm"])
    rows.append(dict(method=method, setting=setting, accuracy=s["acc"], macro_f1=s["macro_f1"], colony_precision=s["colony_p"],
                     colony_recall=s["colony_r"], colony_f1=s["colony_f1"], fold_acc_mean=np.mean(accs), fold_acc_std=np.std(accs)))


# ----------------------------------------------------------------------------- classical features
def stain_map(bgr):
    """Lab colour-space 'stain map' from Mini Project 1: guano is positive in a* and b*, ice is near 0."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = (lab[..., 1] - 128) + (lab[..., 2] - 128)
    return np.clip((s + 5) / 25 * 255, 0, 255).astype(np.uint8)


def channel_image(bgr, channel):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if channel == "gray" else stain_map(bgr)


def sift_descriptors(images, channel):
    # contrastThreshold lowered from the default 0.04: ice is low-contrast, the default finds almost nothing
    sift = cv2.SIFT_create(contrastThreshold=0.01)
    out = []
    for bgr in images:
        _, d = sift.detectAndCompute(channel_image(bgr, channel), None)
        out.append(d.astype(np.float32) if d is not None else np.zeros((0, 128), np.float32))
    return out


def harris_features(gray):
    """Fixed-length description of the Harris corner response (cv2.cornerHarris) of one image."""
    R = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    rmax = float(R.max())
    scale = rmax if rmax > 0 else 1.0
    feats = [np.log10(max(rmax, 1e-12)), R.mean() / scale, R.std() / scale]
    peaks = (R == cv2.dilate(R, np.ones((5, 5), np.uint8))) & (R > 0)        # non-maximum suppression, corners only
    for t in (0.001, 0.005, 0.01, 0.05, 0.1):                                 # corner counts at several strengths
        feats.append(np.log1p((peaks & (R > t * scale)).sum()))
    h, w = R.shape
    grid = np.zeros((4, 4))                                                   # where the strong corners are
    for y_, x_ in zip(*np.nonzero(peaks & (R > 0.01 * scale))):
        grid[y_ * 4 // h, x_ * 4 // w] += 1
    return np.array(feats + list(np.log1p(grid.ravel())), dtype=np.float64)


def svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))


def sift_fit_predict(descs, y, k):
    def fp(tr, te):
        pool = np.vstack([descs[i] for i in tr if len(descs[i])])
        if len(pool) > 40000:
            pool = pool[np.random.RandomState(SEED).choice(len(pool), 40000, replace=False)]
        km = MiniBatchKMeans(n_clusters=min(k, len(pool)), random_state=SEED, n_init=3, batch_size=2048).fit(pool)  # vocabulary: TRAIN descriptors only

        def hist(i):
            h = np.zeros(km.n_clusters)
            if len(descs[i]):
                h = np.bincount(km.predict(descs[i]), minlength=km.n_clusters).astype(float)
                h /= h.sum()
            return h
        clf = svm().fit(np.array([hist(i) for i in tr]), y[tr])
        return clf.predict(np.array([hist(i) for i in te]))
    return fp


def feature_fit_predict(X, y):
    def fp(tr, te):
        return svm().fit(X[tr], y[tr]).predict(X[te])
    return fp


# ----------------------------------------------------------------------------- deep learning
def run_deep(arch, df, folds, y, names, args):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    n = len(names)
    lr = args.lr or (1e-4 if arch == "resnet" else 5e-5)
    print(f"  device: {device} | epochs {args.epochs} | lr {lr} | batch {args.batch}")
    imgs = [Image.open(p).convert("RGB") for p in df.path]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]          # ImageNet statistics, as the pretrained weights expect

    class Rot90Flip:                                                    # geometric augmentation from Mini Project 1
        def __call__(self, im):
            k = random.randint(0, 3)
            if k:
                im = im.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[k])
            return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() < 0.5 else im

    train_tf = T.Compose([Rot90Flip(), T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])

    def batches(idx, tf, shuffle):
        idx = list(idx)
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            if shuffle and len(b) < 2:                                  # BatchNorm cannot train on a single sample
                continue
            yield torch.stack([tf(imgs[j]) for j in b]).to(device), torch.tensor(y[b]).to(device)

    def build():
        if arch == "resnet":
            import torchvision
            try:
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:
                print("  WARNING: could not load pretrained ResNet-18 weights (", e, ") -> random init")
                m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, n)
            return m
        import timm
        try:
            return timm.create_model("mvitv2_tiny", pretrained=True, num_classes=n, drop_path_rate=0.1)
        except Exception as e:
            print("  WARNING: could not load pretrained MViTv2-tiny weights (", e, ") -> random init")
            return timm.create_model("mvitv2_tiny", pretrained=False, num_classes=n)

    def fit_predict(tr, te):
        m = build().to(device)
        counts = np.bincount(y[tr], minlength=n)
        w = torch.tensor(len(tr) / (n * np.maximum(counts, 1)), dtype=torch.float32).to(device)   # class-balanced loss
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for ep in range(args.epochs):
            m.train()
            tot = 0.0
            for x, t in batches(tr, train_tf, True):
                opt.zero_grad()
                loss = loss_fn(m(x), t)
                loss.backward()
                opt.step()
                tot += loss.item() * len(t)
            sched.step()
            if ep in (0, args.epochs - 1) or (ep + 1) % 5 == 0:
                print(f"    epoch {ep + 1:2d}/{args.epochs}  train loss {tot / len(tr):.3f}")
        m.eval()
        pred = []
        with torch.no_grad():
            for x, _ in batches(te, test_tf, False):
                pred += m(x).argmax(1).cpu().tolist()
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        return np.array(pred)

    return cross_validate(folds, y, fit_predict)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset folder (the one containing processed/final); auto-detected if omitted")
    ap.add_argument("--tasks", default="all", help="comma list of: sift, harris, resnet, mvit, all")
    ap.add_argument("--setting", default="both", choices=["both", "three", "binary"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=100, help="visual-vocabulary size for SIFT")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", help="optional CSV file for the summary table")
    args, _ = ap.parse_known_args()      # ignore extra args, e.g. the "-f kernel.json" that Jupyter / Colab adds

    random.seed(SEED)
    np.random.seed(SEED)
    tasks = ["sift", "harris", "resnet", "mvit"] if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    final_dir = find_final_dir(args.data)
    df = load_samples(final_dir)
    folds = make_folds(df, args.folds)

    print(f"dataset: {final_dir}\n{len(df)} tiles | per class: {dict(zip(CLASSES, np.bincount(df.y3, minlength=3)))} | sites: {df.site.nunique()}")
    print(f"majority-class baseline accuracy (3-class): {np.bincount(df.y3).max() / len(df):.3f}  |  (binary): {1 - (df.y3 == COLONY).mean():.3f}")
    for k, (tr, te) in enumerate(folds):
        print(f"  fold {k + 1}: train {len(tr)} / test {len(te)} (colony in test: {(df.y3.values[te] == COLONY).sum()}, sites in test: {df.site.values[te].tolist() and len(set(df.site.values[te]))})")

    images = None
    if any(t in tasks for t in ("sift", "harris")):
        images = [cv2.imread(p) for p in df.path]
    cache, rows = {}, []
    settings = ["three", "binary"] if args.setting == "both" else [args.setting]

    for setting in settings:
        y, names = labels_for(df, setting)
        print(f"\n{'=' * 78}\nSETTING: {setting}   classes: {names}\n{'=' * 78}")
        for task in tasks:
            if task == "sift":
                for ch in ("gray", "stain"):
                    t0 = time.time()
                    if ("sift", ch) not in cache:
                        cache[("sift", ch)] = sift_descriptors(images, ch)
                        counts = [len(d) for d in cache[("sift", ch)]]
                        print(f"  SIFT on {ch}: mean {np.mean(counts):.0f} keypoints/tile, {sum(c == 0 for c in counts)} tiles with none")
                    oof, accs = cross_validate(folds, y, sift_fit_predict(cache[("sift", ch)], y, args.k))
                    report(rows, f"SIFT BoVW + SVM ({ch})", setting, y, oof, accs, names, time.time() - t0)
            elif task == "harris":
                for ch in ("gray", "stain"):
                    t0 = time.time()
                    if ("harris", ch) not in cache:
                        cache[("harris", ch)] = np.array([harris_features(channel_image(b, ch)) for b in images])
                    oof, accs = cross_validate(folds, y, feature_fit_predict(cache[("harris", ch)], y))
                    report(rows, f"Harris + SVM ({ch})", setting, y, oof, accs, names, time.time() - t0)
            elif task in ("resnet", "mvit"):
                try:
                    import torch  # noqa: F401
                    if task == "mvit":
                        import timm  # noqa: F401
                except ImportError as e:
                    print(f"\n[{task}] skipped: {e}. Install torch/torchvision/timm (Colab or Kaggle) to run the deep-learning part.")
                    continue
                t0 = time.time()
                print(f"\n[{task} | {setting}] training {len(folds)} folds ...")
                oof, accs = run_deep(task, df, folds, y, names, args)
                report(rows, "ResNet-18 (fine-tuned)" if task == "resnet" else "MViTv2-tiny (fine-tuned)", setting, y, oof, accs, names, time.time() - t0)
            else:
                print(f"unknown task: {task}")

    if rows:
        table = pd.DataFrame(rows)
        print(f"\n{'=' * 78}\nSUMMARY (pooled out-of-fold predictions; sites never shared between train and test)\n{'=' * 78}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if args.out:
            table.to_csv(args.out, index=False)
            print("saved", args.out)


if __name__ == "__main__":
    main()
