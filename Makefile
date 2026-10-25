.PHONY: install install-full convert train infer eval test lint

install:
	pip install -e ".[dev]"

install-full:
	pip install -r requirements.txt && pip install -e ".[dev]"

convert:
	python scripts/coco_to_yolo_seg.py --overwrite

train:
	python scripts/train_yolov8_seg.py

infer:
	python scripts/infer_realsense.py --weights weights/best.pt

eval:
	python scripts/evaluate.py --pred predictions.jsonl --gt examples/gt_example.json --out reports/latest

test:
	pytest

lint:
	ruff check .
