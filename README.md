# Penguin-colony monitoring from Sentinel-2 satellite images

Image Processing mini projects 1 and 2.
Team: PES1UG24CS435, PES1UG24CS441, PES1UG24CS455, PES1UG24CS472

Emperor/Adelie/Chinstrap penguin colonies in Antarctica are located from space by their **guano stains**: faint
reddish-brown speckles on ice. At Sentinel-2's 10 m/pixel a single penguin is about 1/1000 of a pixel, so the stain is a
weak colour signal rather than a visible object.

## Mini Project 1 - dataset and preprocessing

* `dataset/raw/<group>/IMG_###.png` - unmodified Sentinel-2 L1C true-colour tiles (512 px, 5 km), 124 labelled tiles:
  `colony` 24, `confuser` 44 (cliff edges, rock, glacier), `clean_ice_or_sea` 56. `uncertain` (7) and `rejected` tiles are kept
  separately and not used for classification.
* `dataset/metadata.csv` - one row per tile (location, date, source, notes).
* `dataset/processed/final/<group>/` - tiles after the final pipeline (crop no-data, white balance, CLAHE on Lab-L, resize 256).
* `dataset/review/` - labelling sheets and the label files.
* `report/Preprocessing_Report.pdf` - the techniques used, with before/after pairs and justification.
* `scripts/` - `download_tiles.py` (Copernicus Data Space; needs `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`), `find_dates.py`,
  `preprocess.py`, `make_figures.py`, `build_report.py`, `review.py`, `build_package.py`, `build_remaining.py`.
  The reference CSVs in the root (`colonies_reference.csv`, `peninsula_colonies_reference.csv`, `tile_plan*.csv`) drive tile selection.

Re-create the demo/normalised preprocessing outputs: `python3 scripts/preprocess.py`.

## Mini Project 2 - classification

Single file: `PES1UG24CS435_PES1UG24CS441_PES1UG24CS455_PES1UG24CS472.py`

| Part | Method |
|---|---|
| 1 | SIFT -> bag of visual words (k=100) -> RBF SVM; Harris corner statistics -> RBF SVM (each on grayscale and on the Lab "stain map") |
| 2 | ResNet-18 and MViTv2-tiny, ImageNet-pretrained, fine-tuned (rot90/flip augmentation, class-weighted loss) |

Evaluated in two settings, `three` (colony / confuser / clean ice) and `binary` (colony vs other), with 5-fold
StratifiedGroupKFold **grouped by colony site** (tiles of the same site on different dates are near-duplicates; a random
split would leak). Metrics are pooled out-of-fold macro-F1 and colony precision/recall/F1; accuracy is misleading because
the binary majority baseline is already about 0.81.

```bash
pip install numpy pandas scikit-learn opencv-python          # classical part
python PES1UG24CS435_PES1UG24CS441_PES1UG24CS455_PES1UG24CS472.py --data dataset --tasks sift,harris

pip install torch torchvision timm pillow                     # deep part (use a GPU: Colab / Kaggle)
python PES1UG24CS435_PES1UG24CS441_PES1UG24CS455_PES1UG24CS472.py --data dataset --tasks resnet,mvit --out results.csv
```

### Results (124 tiles, 24 colonies; single seed, so differences of a few hundredths are within noise)

| Method | Setting | Accuracy | Macro-F1 | Colony F1 |
|---|---|---|---|---|
| SIFT BoVW + SVM (gray) | 3-class | 0.524 | 0.456 | 0.216 |
| Harris + SVM (stain) | 3-class | 0.411 | 0.313 | 0.000 |
| ResNet-18 | 3-class | 0.589 | 0.533 | 0.326 |
| MViTv2-tiny | 3-class | 0.556 | 0.495 | 0.222 |
| SIFT BoVW + SVM (stain) | binary | 0.790 | 0.534 | 0.188 |
| Harris + SVM (stain) | binary | 0.798 | 0.563 | 0.242 |
| ResNet-18 | binary | 0.661 | 0.474 | 0.160 |
| MViTv2-tiny | binary | 0.774 | 0.599 | 0.333 |

Full deep-learning results are in `results.csv`. All methods are weak: the guano signal is tiny and there are only 24 colony tiles.
