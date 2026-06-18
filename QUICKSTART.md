# FireVision: exact runbook

Run every command from the repository root.

## 1. Install the complete environment

Recommended:

```bash
cd /path/to/Computer-vision
bash scripts/setup_env.sh .venv-full
source .venv-full/bin/activate
python scripts/doctor.py
```

The setup uses Python 3.12 by default and installs PyTorch, TorchVision, OpenCV, Ultralytics,
segmentation-models-pytorch, plotting tools, Gradio, and test tools.

If Python 3.12 is unavailable:

```bash
PYTHON_BIN=python3.11 bash scripts/setup_env.sh .venv-full
```

For NVIDIA systems, if PyTorch reports CPU instead of CUDA, install the correct PyTorch wheel from
the official PyTorch selector, then rerun `python scripts/doctor.py`.

## 2. Download and prepare D-Fire

Use the official D-Fire download linked by:

```bash
python scripts/list_datasets.py
```

Assuming the extracted images and labels are `/datasets/D-Fire/images` and
`/datasets/D-Fire/labels`:

```bash
python scripts/prepare_dfire.py \
  --root /datasets/D-Fire \
  --output data/manifests/dfire.csv

python scripts/check_manifest.py --manifest data/manifests/dfire.csv

python scripts/build_yolo_dataset.py \
  --manifest data/manifests/dfire.csv \
  --output data/yolo/dfire
```

Inspect `data/yolo/dfire/data.yaml`. D-Fire defaults are `0=smoke`, `1=fire`.

## 3. Train a fast baseline

Start small to verify the whole pipeline:

```bash
python scripts/train_detector.py \
  --data data/yolo/dfire/data.yaml \
  --model yolo11n.pt \
  --epochs 10 \
  --batch-size 8 \
  --output runs/yolo11n_smoke_test
```

Then run a proper experiment:

```bash
python scripts/train_detector.py \
  --data data/yolo/dfire/data.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --batch-size 16 \
  --output runs/yolo11s
```

On memory errors, halve `--batch-size`. For CPU training, begin with `yolo11n.pt`.

## 4. See detections

Annotated images:

```bash
python scripts/infer_images.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --source "/path/to/test/images" \
  --output runs/examples
```

Open `runs/examples/annotated/`.

Annotated video with persistence alerts:

```bash
python scripts/infer_video.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --source /path/to/test.mp4 \
  --confidence 0.25 \
  --window 10 \
  --required 6 \
  --output runs/video/events.csv \
  --output-video runs/video/annotated.mp4 \
  --frame-log runs/video/frame_predictions.csv
```

Open `runs/video/annotated.mp4`.

Browser UI:

```bash
python scripts/demo.py \
  --weights runs/yolo11s/train/weights/best.pt
```

Visit <http://127.0.0.1:7860>, upload an image, and move the confidence slider.

## 5. Measure TP/TN/FP/FN and plot results

Create image-level predictions from the detector:

```bash
python scripts/evaluate_detector_presence.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --manifest data/manifests/dfire.csv \
  --split val \
  --output runs/yolo11s/presence_val

python scripts/evaluate_detector_presence.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --manifest data/manifests/dfire.csv \
  --split test \
  --batch-size 4 \
  --device 0 \
  --output runs/yolo11s/presence
```

The evaluator deliberately loads only a few images at a time. If it reports CUDA out-of-memory,
rerun with `--batch-size 1`.

Tune thresholds using validation predictions, never test predictions:

```bash
python scripts/tune_thresholds.py \
  --predictions runs/yolo11s/presence_val/predictions.csv \
  --minimum-specificity 0.95 \
  --output runs/classifier/thresholds.json
```

Create interactive confusion matrices, ROC curves, precision-recall curves, and score distributions:

```bash
python scripts/visualize_metrics.py \
  --predictions runs/yolo11s/presence/predictions.csv \
  --threshold 0.25 \
  --output runs/yolo11s/plots
```

Open `runs/yolo11s/plots/fire_evaluation.html` and `smoke_evaluation.html` in a browser.

## 6. Train classification and temporal models

```bash
python scripts/train_classifier.py \
  --manifest data/manifests/dfire.csv \
  --model resnet18 \
  --epochs 20 \
  --output runs/resnet18
```

For temporal training, first create and label frames from source videos:

```bash
python scripts/extract_video_frames.py \
  --videos /datasets/fire-videos \
  --output data/frames \
  --fps 2 \
  --manifest data/manifests/video_frames.csv
```

After adding correct `fire` and `smoke` labels:

```bash
python scripts/split_manifest.py \
  --input data/manifests/video_frames_labeled.csv \
  --output data/manifests/video_frames_split.csv

python scripts/check_manifest.py \
  --manifest data/manifests/video_frames_split.csv

python scripts/train_temporal.py \
  --manifest data/manifests/video_frames_split.csv \
  --model cnn_lstm \
  --sequence-length 16 \
  --stride 2 \
  --batch-size 4 \
  --output runs/cnn_lstm
```

Only use Video Swin after CNN–LSTM works; it is substantially heavier:

```bash
python scripts/train_temporal.py \
  --manifest data/manifests/video_frames_split.csv \
  --model video_swin_t \
  --sequence-length 16 \
  --batch-size 1 \
  --output runs/video_swin
```

## 7. Robustness checklist

- Keep complete videos/events/cameras in one split.
- Keep the test split untouched until the final report.
- Add fog, steam, cloud, dust, exhaust, glare, sunset, red lights, and welding as negatives.
- Report fire and smoke separately.
- Test on a dataset and camera not used during training.
- Measure false alerts per camera-hour and detection delay for video.
- Tune confidence and persistence on validation data only.
- Keep visual detection as one signal alongside certified smoke/heat sensors for safety use.
