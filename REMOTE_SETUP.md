# VS Code Remote SSH setup

Everything below runs on the remote server, not on your laptop.

## 1. Open the remote terminal

In VS Code:

1. Connect using **Remote - SSH**.
2. Open the remote project folder.
3. Select **Terminal → New Terminal**.
4. Confirm the terminal is remote:

```bash
hostname
pwd
```

## 2. Get the latest source code

If the repository is not cloned:

```bash
git clone https://github.com/Johnzyprogramador/Computer-vision.git
cd Computer-vision
```

If it is already cloned:

```bash
cd /path/to/Computer-vision
git pull origin main
```

Do not place D-Fire inside Git history. The repository ignores `data/`, model weights, environments,
and generated runs.

## 3. Check server prerequisites

```bash
python3 --version
nvidia-smi || true
```

The installer supports Python 3.10, 3.11, and 3.12. If Ubuntu lacks the `venv` module:

```bash
sudo apt update
sudo apt install python3-venv
```

If you cannot use `sudo`, ask the server administrator to provide Python with `venv`, or use an
existing Conda environment.

## 4. Install the complete environment remotely

```bash
bash scripts/setup_env.sh .venv-full
source .venv-full/bin/activate
python scripts/doctor.py
```

Expected output includes installed versions for PyTorch, TorchVision, OpenCV, Ultralytics,
segmentation-models-pytorch, Matplotlib, and Gradio.

For an NVIDIA server, `Training device` should show `CUDA (...)`. If it says CPU despite
`nvidia-smi` working, install the PyTorch build recommended by the official PyTorch selector inside
the active environment, then rerun:

```bash
python scripts/doctor.py
```

## 5. Select the environment in VS Code

Use **Python: Select Interpreter** and choose:

```text
/path/to/Computer-vision/.venv-full/bin/python
```

New VS Code terminals may activate it automatically. Otherwise:

```bash
source .venv-full/bin/activate
```

## 6. Locate D-Fire

Suppose the uploaded dataset is:

```text
/remote/datasets/D-Fire/
├── images/
└── labels/
```

Confirm it:

```bash
find /remote/datasets/D-Fire/images -type f | head
find /remote/datasets/D-Fire/labels -type f | head
du -sh /remote/datasets/D-Fire
```

Prepare the benchmark:

```bash
python scripts/prepare_dfire.py \
  --root /remote/datasets/D-Fire \
  --output data/manifests/dfire.csv

python scripts/check_manifest.py \
  --manifest data/manifests/dfire.csv

python scripts/build_yolo_dataset.py \
  --manifest data/manifests/dfire.csv \
  --output data/yolo/dfire
```

D-Fire defaults are `0=smoke`, `1=fire`. Inspect several labels and the downloaded metadata before
starting a long run.

## 7. Verify the complete installation

```bash
pytest -q
ruff check src scripts tests

python scripts/train_detector.py --help
python scripts/train_classifier.py --help
python scripts/train_temporal.py --help
python scripts/train_segmenter.py --help
python scripts/demo.py --help
```

## 8. Start with a short training run

```bash
python scripts/train_detector.py \
  --data data/yolo/dfire/data.yaml \
  --model yolo11n.pt \
  --epochs 3 \
  --batch-size 8 \
  --device 0 \
  --output runs/yolo_smoke_test
```

If there is no NVIDIA GPU, omit `--device 0`. If GPU memory is exhausted, reduce `--batch-size` to
4, 2, or 1.

After the smoke test succeeds:

```bash
python scripts/train_detector.py \
  --data data/yolo/dfire/data.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --batch-size 16 \
  --device 0 \
  --output runs/yolo11s
```

Use `tmux` or `screen` for long training so it survives an SSH disconnect:

```bash
tmux new -s firevision
source .venv-full/bin/activate
# Run training here.
```

Detach with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux attach -t firevision
```

## 9. View the remote Gradio demo through SSH

On the remote server:

```bash
source .venv-full/bin/activate
python scripts/demo.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --host 127.0.0.1 \
  --port 7860
```

In VS Code, open the **Ports** panel and forward remote port `7860`. Then visit
<http://127.0.0.1:7860> on your laptop. The browser is local, but inference and model files stay on
the remote server.

## 10. View generated files

Annotated images:

```bash
python scripts/infer_images.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --source /remote/path/to/test/images \
  --output runs/examples
```

Use VS Code's remote Explorer to open/download files below `runs/examples/annotated/`.

Interactive metric reports:

```bash
python scripts/evaluate_detector_presence.py \
  --weights runs/yolo11s/train/weights/best.pt \
  --manifest data/manifests/dfire.csv \
  --split test \
  --output runs/yolo11s/presence

python scripts/visualize_metrics.py \
  --predictions runs/yolo11s/presence/predictions.csv \
  --threshold 0.25 \
  --output runs/yolo11s/plots
```

Open or download the generated HTML files through VS Code's remote Explorer.

