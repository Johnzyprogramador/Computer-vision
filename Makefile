.PHONY: setup doctor test datasets demo

setup:
	bash scripts/setup_env.sh .venv-full

doctor:
	.venv-full/bin/python scripts/doctor.py

test:
	.venv-full/bin/pytest -q
	.venv-full/bin/ruff check src scripts tests

datasets:
	.venv-full/bin/python scripts/list_datasets.py

demo:
	@test -n "$(WEIGHTS)" || (echo "Usage: make demo WEIGHTS=runs/yolo11n/train/weights/best.pt"; exit 1)
	.venv-full/bin/python scripts/demo.py --weights "$(WEIGHTS)"
