# Early-fire dataset integration

Run `python scripts/inspect_dataset.py --root /path/to/extracted/dataset` before preparation. It
prints the real folder tree and counts without changing files.

## PYRONEAR-2025

PYRONEAR is the highest-priority external benchmark for early wildfire smoke. The paper reports
about 50,000 images, 150,000 manual annotations, 640 fires, and image/video data from France, Spain,
Chile, and the United States.

The image branch uses YOLO datasets. After downloading the official archive, locate its `data.yaml`
and run:

```bash
python scripts/prepare_yolo_dataset.py \
  --data /datasets/PYRONEAR/data.yaml \
  --dataset-name pyronear_2025 \
  --output data/manifests/pyronear_2025.csv

python scripts/check_manifest.py \
  --manifest data/manifests/pyronear_2025.csv

python scripts/build_yolo_dataset.py \
  --manifest data/manifests/pyronear_2025.csv \
  --output data/yolo/pyronear_2025
```

The adapter reads class names from the source YAML and remaps them to FireVision's canonical
`0=smoke, 1=fire`. It accepts directory-based splits and `.txt` image lists. It refuses to invent a
test set if the source archive does not provide one.

For a clean cross-dataset result, first test the D-Fire-trained YOLO checkpoint on PYRONEAR without
fine-tuning. Then train on PYRONEAR and test on both datasets.

## FIgLib / SmokeyNet

FIgLib is specifically designed around initial wildfire smoke observed by fixed cameras. Use the
official `smokeynet_train.json`, `smokeynet_valid.json`, and `smokeynet_test.json` files:

```bash
python scripts/prepare_figlib.py \
  --images-root /datasets/FIgLib/images \
  --train-json /datasets/FIgLib/smokeynet_train.json \
  --val-json /datasets/FIgLib/smokeynet_valid.json \
  --test-json /datasets/FIgLib/smokeynet_test.json \
  --output data/manifests/figlib.csv
```

The adapter:

- matches JSON keys to extracted images;
- retains official train/validation/test splits;
- groups frames by event/camera;
- sets `frame_index` from timestamps;
- rejects missing images and event leakage.

It can feed the image classifiers or temporal training:

```bash
python scripts/train_temporal.py \
  --manifest data/manifests/figlib.csv \
  --model cnn_lstm \
  --sequence-length 16 \
  --stride 1 \
  --label-mode last \
  --output runs/figlib_cnn_lstm
```

`--label-mode last` is intentional for early detection: the prediction target is the state at the
end of the observed sequence. `max` remains available for datasets whose event definition is
"smoke occurred anywhere in this clip."

## SmokeBench

SmokeBench contains paired early-stage smoky and clean surveillance images. It is a restoration
dataset rather than a bounding-box detector dataset, but the pairs are useful for smoke
classification and robustness.

After identifying the two paired folders:

```bash
python scripts/prepare_smokebench.py \
  --smoky-dir /datasets/SmokeBench/train/smoke \
  --clean-dir /datasets/SmokeBench/train/clear \
  --output data/manifests/smokebench.csv
```

Folder names vary by release; use `inspect_dataset.py` and pass the actual paths. The script requires
matching relative filenames and keeps each smoky/clean pair in the same split to prevent leakage.

## FireSentry

FireSentry is not an early-ignition classification benchmark. It targets fine-grained fire spread
forecasting. Region A currently contains paired:

```text
Region A/
├── Infrared Videos/
├── Fire Mask Videos/
├── Visible Light/
└── Environmental Info/
```

Prepare infrared segmentation frames:

```bash
python scripts/prepare_firesentry.py \
  --region "/datasets/FireSentry/Region A" \
  --output-root data/firesentry_region_a \
  --manifest data/manifests/firesentry_masks.csv \
  --every 10
```

Then use `train_segmenter.py`. Whole videos remain in one split.

## What the recent papers do

### PYRONEAR-2025

The authors benchmark a lightweight YOLO detector on individual frames and a sequential model for
video. The temporal branch uses a frozen ResNet feature extractor followed by an LSTM. Its purpose
is not merely higher frame accuracy: it combines recent observations to improve recall and detect
smoke earlier while maintaining precision. Their cross-dataset experiments are especially relevant
because performance is much lower than on easy in-domain datasets.

### FireSentry / FiReDiff

FireSentry solves a different problem: forecasting how a known fire evolves. FiReDiff first predicts
future infrared video frames using video-generation models, then obtains fire masks from the
predicted infrared frames using SAM2. The paper evaluates generated-video quality with PSNR, SSIM,
LPIPS, and FVD, and mask quality with AUPRC, F1, IoU, and MSE.

### GWFP

The June 2026 GWFP paper describes a geographically diverse RGB/video/NIR dataset with smoke, flame,
embers, fog/waterdog, and hard negatives. It benchmarks convolutional and transformer architectures
and proposes an HTE-ResNet variant that combines frequency and spatial features through
Hadamard-enhanced residual interactions. However, the authors state that the dataset and source code
will be released upon acceptance. It should not yet be treated as a downloadable benchmark.

