# FireVision Bench

A reproducible benchmark for visible fire and smoke detection. It supports:

- image classification: ResNet-18 and EfficientNet-B0;
- object detection: Ultralytics YOLO (YOLOv8/YOLO11 checkpoints);
- pixel segmentation: U-Net and DeepLabV3+;
- temporal video models: CNN–LSTM and Video Swin Transformer;
- persistence-filtered video alerts;
- TP, TN, FP, FN, accuracy, precision, recall/sensitivity, specificity, F1, IoU,
  false-positive rate, false-negative rate, and balanced accuracy.

This is research software, not a certified life-safety alarm. Never use vision as the only fire
detection channel in a safety-critical installation.

For a literal installation → training → visualization walkthrough, use
[`QUICKSTART.md`](QUICKSTART.md).
For a server accessed through VS Code Remote SSH, use
[`REMOTE_SETUP.md`](REMOTE_SETUP.md).

## Why the data format matters

All image/video experiments use a CSV manifest:

```csv
path,group_id,frame_index,split,fire,smoke
/data/cam1/0001.jpg,cam1_event7,1,train,0,1
/data/cam1/0002.jpg,cam1_event7,2,train,0,1
```

`group_id` is the original video, event, camera burst, or scene. A group must occur in exactly one
split. This prevents nearly identical adjacent frames from leaking into validation/test data.

Segmentation uses `image,mask,group_id,split`. Train one binary mask model per target (`fire` or
`smoke`) unless the source dataset genuinely supplies separate class masks.

## Install

```bash
bash scripts/setup_env.sh .venv-full
source .venv-full/bin/activate
python scripts/doctor.py
```

This uses Python 3.12 and installs the full training and visualization stack. For a smaller manual
install, use `.[train]`, `.[detection]`, `.[segmentation]`, or `.[visualization]`.

## Dataset workflow

List curated primary sources:

```bash
python scripts/list_datasets.py
```

Downloads are deliberately manual: several datasets require accepting terms, use cloud-hosted
archives, or do not state a redistribution license clearly. The catalog is
[`configs/datasets.yaml`](configs/datasets.yaml).

### D-Fire

Download D-Fire from its official repository, then:

```bash
python scripts/prepare_dfire.py \
  --root /datasets/D-Fire \
  --images images \
  --labels labels \
  --output data/manifests/dfire.csv

python scripts/check_manifest.py --manifest data/manifests/dfire.csv
```

D-Fire class IDs default to `0=smoke`, `1=fire`. Verify this against the downloaded edition and use
`--smoke-class-id` / `--fire-class-id` if its metadata differs.
The script recognizes flat `images/` + `labels/` and pre-split `train/` + `test/` layouts. For the
latter, the official test set is preserved and 15% of train is deterministically reserved for
validation when no `val/` directory exists.

For YOLO:

```bash
python scripts/build_yolo_dataset.py \
  --manifest data/manifests/dfire.csv \
  --output data/yolo/dfire
```

### Video datasets

```bash
python scripts/extract_video_frames.py \
  --videos /datasets/fire-videos \
  --output data/frames \
  --fps 2 \
  --manifest data/manifests/video_frames.csv
```

Review/add labels, then assign whole-group splits:

```bash
python scripts/split_manifest.py \
  --input data/manifests/video_frames_labeled.csv \
  --output data/manifests/video_frames_split.csv
```

Do not randomly split frames. Do not let overlapping temporal windows cross splits. Prefer splitting
by physical camera as well as event when measuring geographic/camera generalization.

## Train and test every model family

### Image classifiers

```bash
python scripts/train_classifier.py --manifest data/manifests/dfire.csv --model resnet18
python scripts/train_classifier.py --manifest data/manifests/dfire.csv \
  --model efficientnet_b0 --output runs/efficientnet
```

### YOLO object detection

```bash
python scripts/train_detector.py \
  --data data/yolo/dfire/data.yaml \
  --model yolo11n.pt \
  --output runs/yolo11n

python scripts/evaluate_detector_presence.py \
  --weights runs/yolo11n/train/weights/best.pt \
  --manifest data/manifests/dfire.csv \
  --output runs/yolo11n/presence
```

YOLO validation reports mAP, box precision, and box recall. Object-level TN is not mathematically
well-defined because background contains infinitely many possible boxes. The second command evaluates
image-level class presence, where TP/TN/FP/FN are meaningful.

### Segmentation

```bash
python scripts/train_segmenter.py \
  --manifest data/manifests/fire_masks.csv \
  --architecture unet \
  --output runs/unet_fire

python scripts/train_segmenter.py \
  --manifest data/manifests/smoke_masks.csv \
  --architecture deeplabv3plus \
  --output runs/deeplab_smoke
```

The reported confusion counts are pixel-level. For large datasets these counts can be huge; F1 and
IoU are usually easier to compare.

### Temporal models

```bash
python scripts/train_temporal.py \
  --manifest data/manifests/video_frames_split.csv \
  --model cnn_lstm \
  --sequence-length 16 \
  --stride 2 \
  --output runs/cnn_lstm

python scripts/train_temporal.py \
  --manifest data/manifests/video_frames_split.csv \
  --model video_swin_t \
  --sequence-length 16 \
  --stride 2 \
  --output runs/video_swin
```

Temporal labels are the class-wise maximum over a clip. Change this only if the benchmark defines a
different event rule. Start with frozen/low-rate pretrained backbones for small video datasets,
monitor event-level rather than only frame-level performance, and tune alert thresholds only on
validation—not test.

### Persistent alerts

```bash
python scripts/infer_video.py \
  --weights runs/yolo11n/train/weights/best.pt \
  --source camera_test.mp4 \
  --window 10 \
  --required 6 \
  --output runs/inference/events.csv \
  --output-video runs/inference/annotated.mp4
```

### Visualize examples and metrics

```bash
python scripts/infer_images.py \
  --weights runs/yolo11n/train/weights/best.pt \
  --source /path/to/example/images

python scripts/demo.py \
  --weights runs/yolo11n/train/weights/best.pt

python scripts/visualize_metrics.py \
  --predictions runs/yolo11n/presence/predictions.csv \
  --threshold 0.25 \
  --output runs/yolo11n/plots
```

The demo is available at `http://127.0.0.1:7860`. The plotting command creates self-contained,
interactive HTML reports with confusion matrices, ROC and precision-recall curves, and confidence
distributions.

### Compare runs

```bash
python scripts/compare_runs.py \
  runs/classifier/test_metrics.json \
  runs/cnn_lstm/test_metrics.json \
  runs/yolo11n/presence/metrics.json
```

## Recommended evaluation protocol

1. Keep one untouched test set per dataset.
2. Report per-class metrics, not only a combined “hazard” class.
3. Include hard negatives: fog, cloud, steam, dust, exhaust, glare, sunsets, red lights.
4. Run cross-dataset tests (train on D-Fire, test on another source) to expose domain shift.
5. For video, report event detection rate, false alarms per camera-hour, and detection delay in
   addition to frame metrics.
6. Save the exact manifest, model checkpoint, threshold, package versions, and random seed.

## Dataset notes

- **D-Fire**: 21,000+ images, YOLO fire/smoke boxes, and a large negative subset.
- **FIgLib**: nearly 25,000 fixed-camera wildfire images and a temporal SmokeyNet benchmark.
- **Corsican Fire Database**: drone wildfire video used for fire segmentation research.
- **FIRESENSE records**: multimodal indoor/outdoor fire video material.
- **SmokeBench**: paired early-stage smoky/clear scenes; useful for robustness or preprocessing,
  not as a substitute for a detection benchmark.

Treat mirrors as untrusted until provenance, annotation mapping, duplicates, and license are checked.
The newest GWFP dataset described in June 2026 was announced as “to be released upon acceptance,” so
it is not included as downloadable data yet.
