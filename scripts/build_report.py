"""Build report/Preprocessing_Report.pdf from the dataset and figures.   python3 scripts/build_report.py
Re-run after `review.py apply` so the group counts in the report are final."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
ss = getSampleStyleSheet()
T = ParagraphStyle("T", parent=ss["Title"], fontSize=20, leading=24)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=13.5)
S = ParagraphStyle("S", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
P = lambda t, s=B: Paragraph(t, s)

# ---- dataset facts (read from the files so the report never disagrees with the data) ----
meta = pd.read_csv(ROOT / "dataset" / "metadata.csv")
lab_path = ROOT / "dataset" / "review" / "labels.csv"
if lab_path.exists():                                     # labels not yet applied: use the review file
    lab = pd.read_csv(lab_path, dtype=str).fillna("")
    counts = lab.new_group.replace("", "not yet reviewed").value_counts()
else:
    counts = meta.group.value_counts()
stats = json.loads((ROOT / "dataset" / "processed" / "normalization_stats.json").read_text())
sizes = meta.groupby(["width", "height"]).size()
n_final = stats["tiles"]
rej_csv = ROOT / "dataset" / "rejected" / "rejected.csv"
n_rej = len(pd.read_csv(rej_csv)) if rej_csv.exists() else 0

TECH = [
 ("01_crop_valid", "1. Cropping", "Input IMG_045; the output keeps only real pixels",
  "Some tiles sit at the edge of a satellite pass, so a black triangle of no-data covers part of them (13 to 33% of several tiles, "
  "up to 89% of a few). Black no-data pixels are not measurements: they would distort brightness statistics, histograms and every filter "
  "that looks at a pixel's neighbours. We keep the largest rectangle that contains only real pixels (after a 3x3 erosion to drop ragged edge pixels).",
  "Largest axis-aligned all-valid rectangle. IMG_045 goes from 512x512 to 272x352 with no black pixels left."),
 ("02_resize", "2. Resizing", "Input IMG_007 at 512x512; output 256x256",
  "Tiles come in two sizes (512x512, plus four older 256x256 tiles) and cropping changes sizes further. Batch processing and any later model need one fixed size, "
  "and 256x256 uses a quarter of the memory. INTER_AREA averages the source pixels when shrinking, which avoids the jagged aliasing that plain subsampling causes.",
  "Final size 256x256, cv2.INTER_AREA. Trade-off: the stain is about half as wide in pixels afterwards. If a later stage needs fine detail, keep 512x512."),
 ("03_white_balance", "3. Colour balance", "Input IMG_007 (raw); output has neutral ice",
  "The tiles are top-of-atmosphere imagery and the ice has a pink-lavender cast (about RGB 213, 201, 218). Guano is a colour signal (reddish-brown), "
  "so judging it against a tinted background is hard, and the cast differs between dates. Ice and snow should be neutral white, so we measure the colour of the bright but "
  "unsaturated pixels (the ice) and scale each channel so that colour becomes grey. The brightest pixels are ignored because they are clipped to white in every channel and hide the cast.",
  "Gains clipped to 0.8 to 1.3 so a dark scene (open water) cannot be pushed to odd colours. IMG_007 ice: (213, 201, 218) before, (209, 210, 210) after."),
 ("04_contrast_stretch", "4. Contrast stretching", "Input IMG_056 (colour-balanced)",
  "Most tiles use only a narrow band of the brightness range, because ice sits near white. Stretching brightness from its 1st to 99.5th percentile to 0 to 255 uses the full range "
  "and exposes faint structure. The same gain is applied to all three channels. In our tests stretching each channel separately turned neutral shadows reddish, "
  "which would create a fake stain.",
  "Percentiles 1 and 99.5 of brightness, one linear transform for R, G and B."),
 ("05_gamma", "5. Gamma correction (intensity transform)", "Input IMG_007 (colour-balanced)",
  "Ice is close to the top of the brightness scale, and a faint stain differs from it by only a few levels. A power-law transform with gamma above 1 spreads out the bright end, "
  "so small differences near white become larger differences. The cost is that the whole picture gets darker.",
  "output = 255 x (input/255)^1.8, applied with a lookup table."),
 ("06_hist_equalize", "6. Global histogram equalisation (tested, not used)", "Input IMG_007 (raw)",
  "Histogram equalisation redistributes brightness so every level is used equally. On a tile that is mostly near-white ice, this exaggerates texture and turns shadows nearly black, "
  "producing features that are not there. We show it to justify choosing the adaptive method below.",
  "Equalisation of the Lab lightness channel only, so colours are untouched."),
 ("07_clahe", "7. CLAHE (contrast enhancement, used)", "Input IMG_007 (colour-balanced)",
  "CLAHE equalises the histogram inside small 8x8 blocks instead of over the whole image, and a clip limit stops it from amplifying noise. Local contrast improves around icebergs and stains "
  "while flat ice stays calm. It works on the lightness channel of Lab, so the colours that identify the stain stay as they were.",
  "clipLimit 2.0, tileGridSize 8x8, on L of Lab."),
 ("08_denoise_median", "8. Noise removal: median filter (tested, not used)", "Input IMG_007 (colour-balanced), 4x zoom on the stain",
  "A 3x3 median replaces each pixel by the median of its neighbourhood. It removes isolated noisy pixels and, unlike a Gaussian blur, keeps edges such as iceberg outlines sharp. "
  "These tiles have little noise, so the effect is subtle. A guano speckle is only two or three pixels wide, and a median filter can erase features that small, "
  "so we leave it out of the final pipeline and note it as a risk to small stains.",
  "3x3 kernel, cv2.medianBlur."),
 ("09_grayscale", "9. Grayscale conversion", "Input IMG_007 (colour-balanced)",
  "Many image-processing methods (edges, thresholding) work on one channel. We show the conversion mainly to make a point: in grayscale the stain becomes a dim grey smudge much like a shadow, "
  "because the information that separates guano from shadow is colour, not brightness. This is also why plain brightness thresholds confuse ice with cloud.",
  "cv2.cvtColor BGR to GRAY on the colour-balanced image."),
 ("10_lab_stain_map", "10. Colour-space conversion: Lab stain map", "Input IMG_007 (colour-balanced)",
  "Lab separates brightness (L) from colour (a*: green to red, b*: blue to yellow). Neutral ice has a* and b* near zero, while guano is reddish-brown, so the sum a* + b* is positive there. "
  "Shadows are dark but neutral, so they disappear in this map and the stain stands out. The scale is fixed, not stretched per image, so tiles with no stain stay dark instead of showing manufactured structure.",
  "stain = (a*-128) + (b*-128), mapped from -5..+20 to 0..255. Edge colour fringes show as faint thin lines (see IMG_004 in the dataset)."),
 ("11_rotate90", "11. Rotation (geometric operation)", "Input IMG_007 (raw)",
  "A stain has no preferred orientation from above, so rotated or flipped copies are equally valid samples. Rotations and flips can multiply the number of colony examples, the scarcest of the three main groups.",
  "np.rot90 (90 degrees); 180, 270 and flips work the same way."),
]


def fig(name, w=15.5 * cm):
    from PIL import Image as PI
    iw, ih = PI.open(FIG / name).size
    return Image(str(FIG / name), width=w, height=w * ih / iw)


doc = SimpleDocTemplate(str(ROOT / "report" / "Preprocessing_Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Preprocessing report: satellite images of penguin colonies")
st = [P("Preprocessing of Satellite Images for Penguin Colony Monitoring", T), P("Mini Project 1: dataset preparation", ss["Heading3"]), Spacer(1, 6)]

st += [P("1. Application and dataset", H1),
       P("Penguin colonies breed in remote parts of Antarctica that are costly and dangerous to survey on the ground. Their guano stains the ice reddish-brown, and the stain is visible from orbit, "
         "so satellite imagery is the practical way to monitor colonies. The images are 5 km square true-colour tiles from the Sentinel-2 satellite (10 m per pixel), "
         "downloaded from the Copernicus Data Space using the published colony locations of Fretwell and Trathan (2021). A single penguin is about 1/1000 of a pixel, so we never see animals, only the stain."),
       P(f"The dataset has <b>{len(meta)} tiles</b> in PNG format. Sizes (width x height): "
         + ", ".join(f"{w}x{h} ({n} tiles)" for (w, h), n in sizes.items()) + ". Group counts:"), Spacer(1, 4)]
tab = Table([["Group", "Tiles"]] + [[k, str(v)] for k, v in counts.items()], colWidths=[7 * cm, 3 * cm])
tab.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF3"))]))
st += [tab, Spacer(1, 4),
       P(f"<b>colony</b>: a reddish-brown stain is visible. <b>confuser</b>: could be mistaken for a stain (dark rock, shadow, cliff edge). <b>clean_ice_or_sea</b>: plain ice, snow or water. "
         f"<b>uncertain</b>: possibly a stain but not confirmable at 10 m. Tiles marked reject ({n_rej}; mostly no-data) were removed and are not in the dataset. Groups were assigned by visual inspection. "
         "Each tile's source, date, location, size and format are recorded in <i>metadata.csv</i>.", S)]

st += [P("2. Why preprocessing is needed here", H1),
       P("The signal we care about is small and faint: a few pixels of brown on a bright, nearly white background, with confusers such as shadows, cliff edges and rock. "
         "The raw tiles also have a colour cast, black no-data regions, mixed sizes, and a brightness range crowded near white. Each technique below targets one of these problems. "
         "Raw tiles are never modified; every output is written to a separate folder."),
       P("Combined pipeline (applied to all tiles): <b>crop no-data, colour balance, CLAHE, resize to 256x256</b>. "
         "The other techniques are shown individually on a set of 12 representative tiles.")]

for fn, title, caption, why, params in TECH:
    st.append(KeepTogether([P(title, H2), P("<b>Why this technique:</b> " + why), Spacer(1, 3), P("<b>Settings and result:</b> " + params, S), Spacer(1, 3), fig(fn + ".png"), P(caption, S)]))

st += [PageBreak(), P("3. Normalisation and final dataset", H1),
       P(f"The {n_final} final tiles are scaled from 0..255 to floating point 0..1 so that later processing is not affected by the integer range. "
         f"We also computed the dataset's per-channel mean and standard deviation (RGB): mean {stats['mean']}, standard deviation {stats['std']}. "
         "A later stage can standardise as (x - mean) / std. The figure shows brightness histograms of one tile before and after the pipeline: "
         "the three colour channels, which were offset from one another by the pink cast, now overlap, and the range is used more fully."),
       fig("histograms.png", 14 * cm),
       P("4. Limitations", H1),
       P("&bull; At 10 m per pixel a stain is only a few pixels, and every step here shrinks or smooths it a little (resizing most of all).<br/>"
         "&bull; Some tiles are ambiguous even after enhancement; those are kept in an <i>uncertain</i> group instead of guessed.<br/>"
         "&bull; The thin colour line along sharp ice edges (IMG_004) can look like a faint stain in the Lab map. Shape and context tell them apart.<br/>"
         "&bull; Labels come from visual inspection, not field confirmation."),
       P("5. Sources", H1),
       P("Imagery: Copernicus Sentinel-2 (Level-1C), accessed through the Copernicus Data Space Ecosystem (free and open). "
         "Colony locations: Fretwell, P. T. and Trathan, P. N. (2021), <i>Discovery of new colonies by Sentinel2 reveals good and bad news for emperor penguins</i>, "
         "Remote Sensing in Ecology and Conservation, doi:10.1002/rse2.176. Peninsula sites: BirdLife International Important Bird Area data as cited on Wikipedia, "
         "and Borowicz et al. (2018), Scientific Reports 8:3926.")]
doc.build(st)
print("report ->", "report/Preprocessing_Report.pdf")
