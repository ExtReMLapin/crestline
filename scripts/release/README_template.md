---
license: cc-by-sa-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
base_model: timm/convnext_base.fb_in22k
tags:
  - mountains
  - alps
  - switzerland
  - skyline
  - ridgeline
  - horizon
  - edge-detection
  - visual-geolocalization
  - dem
  - convnext
  - unet
---

# Crestline — skyline & inner ridgeline detection for mountain photos

## In French: S.D.E.C.E. : Système de Détection de l’Emplacement par Comparaison des Escarpements - nom pris au hasard

**Draws the skyline and the inner ridgelines (a nearer ridge silhouetted against farther terrain) on any mountain photo.**

![Crestline on a photo of the Mont Blanc massif: skyline in green, inner ridgelines in cyan](images/demo.jpg)

## Method

- **Labels come from 3D terrain, not from hand drawing.** We took {{N_TRAIN}} photos from [smapshot](https://smapshot.heig-vd.ch), whose camera poses were aligned by volunteers on a 3D model of Switzerland.
- From each pose we ray-traced a **depth map** in the Swiss elevation model ([swissALTI3D](https://www.swisstopo.admin.ch) 2 m, plus swissALTIRegio far away, with Earth curvature and refraction).
- **Skyline** = where rays stop hitting terrain. **Inner ridgelines** = sharp depth jumps (near terrain occluding terrain ≥ 25 % and ≥ 50 m farther).
- Each pose was refined by a small rotation (≤ 0.6°) to snap terrain lines onto image edges. Photos whose rendered skyline did not match the photo were discarded.
- **Ignored during training:**
  - trees, buildings and people;
  - clouds hiding the terrain;
  - ridges with no visible edge in the photo.
- **Model:** U-Net with a ConvNeXt-Base encoder (ImageNet-22k), 3 outputs (sky, skyline, inner ridges), trained 30 epochs on one RTX 5090 (~25 min).

## Results

Per-column skyline error between the predicted and true sky boundary, in pixels at ~1024 px width.
- **CH1** ({{N_CH1}} photos of the Swiss Alps, skylines corrected by hand, ETH Zurich) is fully independent of our pipeline. It is the reference.
- **smapshot-test** ({{N_SMAP}} photos) comes from geographic areas never seen in training (10 km cells), with labels from our pipeline.

{{TABLE}}

Public segmenters are already good on clear photos (median 1-2 px). They fail on clouds, snow against a white sky, haze and backlight. That is where this model gains most: its mean error is about 2.4× lower than theirs on CH1. **No public model detects inner ridgelines**; this is also where Crestline has the most room to improve.

## Why not GeoPose3K?

We first trained on [GeoPose3K](https://cphoto.fit.vutbr.cz/geoPose3K/) (40 GB, 3,111 photos rendered from a DEM). Its labels are **not precise enough**:
- the DEM is 24-30 m and most poses are semi-automatic, so rendered ridges are often off by tens of pixels, especially on terrain closer than ~3 km. A 30 m DEM cell seen from 500 m is already ~3.4°;
- the same network trained on it is clearly worse (table above), and nearly blind to inner ridges (F1 0.08).

Swapping the label source (hand-aligned poses, 2 m DEM) mattered far more than model size: a ConvNeXt-Large brought no gain.

## Usage

```python
from PIL import Image
import crestline  # crestline.py from this repository

model, config = crestline.load("LPN64/crestline-convnext-base")
image, prob = crestline.predict(model, Image.open("photo.jpg"))       # prob: sky, skyline, inner ridges
skyline_rows, ridges = crestline.lines(prob, config["ridge_threshold"])  # {{RIDGE_THR}}
crestline.overlay(image, skyline_rows, ridges).save("lines.jpg")
```

## More examples (holiday photos, Tour des Fiz, Haute-Savoie)

{{GALLERY}}

## Limitations

- Trained on Swiss Alps photos (1900s to 2020s, many black-and-white). Expect lower quality on very different terrain.
- Inner ridgelines are recall-limited (F1 ≈ 0.35 on smapshot-test): faint ridges in haze are often missed.
- Summits hidden in clouds: the skyline follows the visible boundary, which may be the cloud.

## License and attribution

Weights released under **CC BY-SA 4.0**, because most training photos are CC BY-SA 4.0.

**Training data licences, stated plainly.** The {{N_TRAIN}} training photos are:

| Licence | Photos | Share |
|---|---|---|
| CC BY-SA 4.0 (ETH-Bibliothek Zürich, Bildarchiv) | 3,672 | 58 % |
| swisstopo ("Quelle: swisstopo") | 1,666 | 27 % |
| **No explicit licence in the source metadata** (ETH-Bibliothek Zürich, Bildarchiv) | **918** | **15 %** |
| Public domain | 30 | < 1 % |

The 918 photos without an explicit licence come from the same public archive. ETH-Bibliothek publishes its images under CC BY-SA 4.0 or as public domain, but the licence field of these particular records was empty. **No photo is redistributed in this repository**, only the trained weights. If you are a rights holder and object to this use, please open a discussion on this repository and we will retrain without the concerned images.

Attribution:
- Labels derived from **swissALTI3D / swissALTIRegio © swisstopo**.
- Photos: **ETH-Bibliothek Zürich, Bildarchiv** and **swisstopo**.
- Camera poses: **smapshot** (HEIG-VD, ETH-Bibliothek, swisstopo) and its volunteer contributors.
- Evaluation: CH1 (Baatz, Saurer, Köser, Pollefeys, ETH Zurich). Evaluation only, no image redistributed.
- The holiday photos in this card belong to the repository author.
