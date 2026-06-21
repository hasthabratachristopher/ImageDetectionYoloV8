# Deteksi Kendaraan Beroda Berdasarkan Data Citra Menggunakan YOLOv8


A vehicle detection project using **YOLOv8** to identify and classify different types of wheeled vehicles from image data.

## Overview

This project trains an object detection model to detect and classify five vehicle classes from images: **Ambulans (Ambulance), Bis (Bus), Mobil (Car), Motor (Motorcycle), and Truk (Truck)**. The model is built on the YOLOv8n (nano) architecture using the Ultralytics library, trained on a labeled vehicle detection dataset, and evaluated on validation/test splits.

## Dataset

- **Source**: `VehiclesDetectionDataset` (provided as `Pro_TK1.zip`)
- **Structure**: split into `train`, `valid`, and `test` sets, each with corresponding image and label folders
- **Classes (5)**:
  | Class | Label |
  |---|---|
  | 0 | Ambulans |
  | 1 | Bis |
  | 2 | Mobil |
  | 3 | Motor |
  | 4 | Truk |
- A `data.yaml` file is generated programmatically to point to the train/val/test image paths and define class names for YOLO training.

## Workflow

1. **Import Libraries** — Ultralytics YOLO, OpenCV, PIL, Matplotlib, etc.
2. **Input Data** — unzip and load the vehicle detection dataset
3. **EDA (Exploratory Data Analysis)**:
   - Inspect class names and dataset configuration
   - Check image properties (size, mode, channels)
   - Visualize a random grid of sample training images
4. **Model Training**:
   - Base model: `yolov8n.yaml` initialized with pretrained `yolov8n.pt` weights
   - Training configuration: `epochs=100`, `imgsz=416`, `batch=16`, `lr0=0.001`, `dropout=0.1`
5. **Model Evaluation**:
   - Validation on the test split using the best checkpoint (`best.pt`)
   - Reports precision, recall, mAP50, and mAP50-95 per class
6. **Display Results**:
   - Run inference on sample/test images with bounding box visualization
   - Render an annotated output video in-notebook

## Results

Validation metrics on the test/validation split (sample run):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.645 | 0.532 | 0.598 | 0.447 |
| Ambulans | 0.635 | 0.875 | 0.873 | 0.757 |
| Bis | 0.581 | 0.609 | 0.627 | 0.514 |
| Mobil | 0.652 | 0.346 | 0.450 | 0.315 |
| Motor | 0.637 | 0.565 | 0.622 | 0.359 |
| Truk | 0.720 | 0.267 | 0.420 | 0.289 |

## Tools & Libraries

- `ultralytics` (YOLOv8)
- `torch`, `torchvision`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `numpy`, `pandas`
- Trained on GPU (T4) via Google Colab

## How to Run

```bash
pip install ultralytics opencv-python pillow matplotlib numpy pandas
```

1. Place the dataset (`Pro_TK1.zip`) in the working directory and unzip it.
2. Run the notebook `Project_TK1_Hasthabrata_Christopher_Liatna_2206824741.ipynb` in Jupyter or Google Colab (GPU runtime recommended).
3. The `data.yaml` file will be generated automatically pointing to the dataset paths.
4. Training produces a `best.pt` weights file under `runs/detect/train/weights/`, which is then used for evaluation and inference.

## Output

- Trained weights: `best.pt`
- Evaluation metrics printed per class
- Annotated detection results on sample/test images
- Rendered video with bounding box predictions on vehicles

## License

*(add your license here, e.g. MIT)*
