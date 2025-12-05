## Repository Description

YOLO-Medical-Detector is a Jupyter Notebook-based implementation for medical image object detection using YOLO (You Only Look Once) models. This project focuses on detecting anatomical structures, abnormalities, or medical instruments in X-rays, CT scans, MRIs, and other imaging modalities to support healthcare AI applications.

## Key Features
- Pre-trained YOLO models adapted for medical datasets (e.g., chest X-rays, tumor detection).
- End-to-end notebooks for data preprocessing, model training, inference, and evaluation.
- Support for custom datasets with annotation tools integration.
- Performance metrics visualization (mAP, precision-recall curves).
- GPU-accelerated training compatible with PyTorch and CUDA.

## Quick Start
```bash
git clone https://github.com/Greekkgod/yolo-medical-detector.git
cd yolo-medical-detector
pip install -r requirements.txt
jupyter notebook
```
Run `train.ipynb` for training or `inference.ipynb` for predictions on sample medical images.

## Datasets & Models
- Compatible with public datasets like ChestX-ray14, SIIM-ACR Pneumothorax, or custom COCO-format annotations.
- Weights from YOLOv8/v9 optimized for medical domains.
