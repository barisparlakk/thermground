# HIT-UAV dataset notes

Findings from inspecting the downloaded release. Everything here was measured on the
files unless marked **(unverified)**.

## Provenance

- Release: HIT-UAV v1.2.1, [Zenodo record 7633134](https://zenodo.org/record/7633134),
  `HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip` (814,751,578 bytes, md5 `bb6c5a33659b7507eae83458729c6109`).
- Fetched by `scripts/download_hituav.py` into `data/hituav/`.
- The GitHub repository (`suojiashun/HIT-UAV-Infrared-Thermal-Dataset`) holds labels only, no images.
- Paper: Suo et al., Scientific Data 10, 227 (2023), doi:10.1038/s41597-023-02066-6.
- License: the GitHub README states CC-BY-4.0; the Zenodo v1.2.1 record states CC0. Cite either way.
- Extracted size is about 1.6 GiB. The archive is kept at `data/hituav/*.zip` and can be deleted.

## File layout

The archive extracts to one nested directory, `data/hituav/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/`
(called `<root>` below). The 2,898 images are stored four times, once per label family.

```
<root>/
  normal_xml/    Pascal VOC, axis-aligned boxes
    JPEGImages/      2898 .jpg   (8-bit grayscale)
    Annotations/     2898 .xml
    ImageSets/Main/  train.txt val.txt trainval.txt test.txt   (names without extension)
  rotate_xml/    Pascal VOC-style, oriented boxes (robndbox)
    JPEGImages/      2898 .jpg   (RGB, gray replicated into 3 channels)
    Annotations/     2898 .xml
    ImageSets/Main/  same four split files
  normal_json/   COCO-like, axis-aligned boxes
    train/ val/ test/   images only (2029 / 290 / 579 .jpg, grayscale)
    annotations/        train.json val.json test.json
  rotate_json/   COCO-like, oriented boxes
    train/ val/ test/   images only (grayscale)
    annotations/        train.json val.json test.json log.json
  tools/         authors' conversion and visualization scripts, info/*.txt
  0_readme_images/  README figures
  README.md  LICENSE  CITATION.cff
```

Differences from the GitHub repo: this archive has no `yolo_labels/` and no `video_sample/`,
and adds `ImageSets/`, `annotations/` and the per-split image folders.

## Annotation formats

| Family | Box format | Notes |
|---|---|---|
| `normal_xml` | `xmin ymin xmax ymax` (pixels) | per-object `name`, `pose`, `truncated`, `difficult` |
| `rotate_xml` | `cx cy w h angle` in `<robndbox>` | `pose` is `Unspecified`; `<path>` is the authors' original Windows path |
| `normal_json` | `bbox = [x, y, w, h]`, x/y is the top-left corner | also has `segmentation` (4 corners) and `area`. Annotation list key is `annotation` (singular). |
| `rotate_json` | `bbox = [cx, cy, w, h, angle]` | angle in radians, range 0 to pi (8208 of 17628 train boxes are non-zero). List key is `annotations`. Has a `difficult` flag. |

- The GitHub README says standard boxes are `[xc, yc, w, h]`. The files do not match: `normal_json` is top-left `x, y, w, h`, checked against the XML for the same box.
- The sign and direction convention of the rotation angle is **(unverified)**. The authors' `tools/devtoolkit/visualization.py` rotates points by `-angle`.
- Box counts agree across all four families for every split (train 17,628, val 2,460, test 4,811).
- `rotate_xml` images are the same frames as `normal_xml` but re-encoded: gray replicated to RGB, max pixel difference 1 gray level.

## Splits as provided

| Split | Images | Share | Boxes | Images with no boxes |
|---|---|---|---|---|
| train | 2029 | 70.0% | 17,628 | 21 |
| val | 290 | 10.0% | 2,460 | 3 |
| test | 579 | 20.0% | 4,811 | 8 |

- `trainval.txt` equals train plus val. No image appears in two splits.
- The XML split files and the JSON files define identical splits, for both box families.
- Boxes per image: mean about 8.5, median 6, max 42.

## Classes

| ID | Name | Train | Val | Test | Total |
|---|---|---|---|---|---|
| 0 | Person | 8533 | 1168 | 2611 | 12,312 |
| 1 | Car | 5253 | 719 | 1339 | 7,311 |
| 2 | Bicycle | 3630 | 554 | 796 | 4,980 |
| 3 | OtherVehicle | 102 | 12 | 34 | 148 |
| 4 | DontCare | 110 | 7 | 31 | 148 |

- IDs are identical in both JSON families (`categories`). The XML files use names only.
- `DontCare` is a fifth category that the README does not list. It is intended as an ignore region, and the authors' `category.txt` includes it. 148 boxes in total.
- `OtherVehicle` has only 148 boxes. Per-class results on it will be noisy.
- `difficult` is 1 for 108 boxes and `truncated` is 1 for 1,383 boxes in the XML.

## Flight metadata

Available for every image, in several places.

**Filename** (the most reliable source): `{scenario}_{altitude}_{angle}_{weather}_{frame}`, for example `1_130_90_0_04077.jpg`.

| Field | Values | Meaning |
|---|---|---|
| scenario | 0, 1 | 0 = day, 1 = night (`scenarios` list in `rotate_json`) |
| altitude | 60 to 130, step 10 | metres |
| angle | 30 to 90, step 10 | camera perspective in degrees |
| weather | always 0 | 0 = sunny. `rainy` (1) never occurs. |
| frame | 0 to 10170 | frame index, unique across the dataset |

- **Also stored in:**
  - `rotate_json` `images[]` has `scenario`, `weather`, `perspective`, `altitude` and `date`, all as strings. They match the filename for every train image.
  - `normal_json` `images[]` has only `date_captured`.
  - In `normal_xml`, each object's `<pose>` is the camera angle. It equals the filename angle for every image.
  - In `rotate_xml`, `<folder>` is the capture date, and `<path>` contains day/night, altitude and angle.
- **Counts:** 917 day and 1,981 night images (68% night). There are 105 distinct (scenario, altitude, angle) combinations out of a possible 112.
- **Altitude:** altitude 60 m is the most common, at 863 images across the splits.
- **Capture dates:** 9 distinct dates from 2020-12-17 to 2021-01-23 in train. I did not check the other splits.
- **Unverified:** what angle 90 means physically (I assume straight down). I have not read the paper.
- **Not available:** per-image GPS, camera intrinsics, and temperature or radiometric values.

## Images

- 640 x 512 pixels for all 2,898 images, JPEG.
- Single channel (`L`, 8-bit) in `normal_xml`, `normal_json` and `rotate_json`. The XML `<depth>` is 1.
- `rotate_xml/JPEGImages` is RGB with three identical channels.
- Pixel values are 8-bit intensity (min and max of five sampled images ranged from 1 to 255). They are not calibrated temperatures.

## Things to be aware of

1. **Split leakage.** The split is random per image, and the 2,898 images were sampled from video frames. In the same (scenario, altitude, angle, weather) group, the median gap between consecutive frame ids is 3. 88% of test images have a train or val frame within 5 frame ids, and 99% within 20. Using frame ids as a proxy for temporal adjacency, which I did not confirm by viewing images, near-duplicate scenes are likely to be shared between train and test. Test scores are probably optimistic, and a grouped split by flight condition or date may be needed for honest evaluation.
2. **Weather is constant**, so it carries no signal.
3. **Small objects.** Using the longer side of the rotated boxes, the median is about 25 px, the 5th percentile 14 px, and 11% of non-DontCare boxes are under 16 px.
4. **Label quality is unaudited.** The one `rotate_xml` file I opened carries `verified="no"`; I did not check the rest. The authors' `log.json` reports no missing XML and no bad boxes or categories. I did not review boxes visually.
5. **Nested extraction path** includes a commit hash. Code should not hard-code it.
