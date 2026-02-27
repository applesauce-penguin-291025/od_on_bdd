# BDD100K Object Detection — Data Analysis, Training & Evaluation

## Overview

This project covers two main areas:

1. **Data Analysis** — Exploratory data analysis (EDA) and an interactive dashboard for the BDD100K dataset.
2. **Model Training & Evaluation** — Training and evaluating a YOLO26m object detector on BDD100K.

## Project Structure

```
/app/
├── bdd100k_images_100k/      # Raw dataset (mounted at runtime)
├── data_analysis/
│   ├── EDA.py                # Exploratory data analysis script
│   └── dashboard.py          # Streamlit dashboard
├── docs/
│   ├── data_analysis_report.md
│   └── YOLO26m_Evaluation_Report.md
├── model/
│   ├── dummy_train/          # Minimal training demo (CPU)
│   │   ├── dataloader.py
│   │   ├── model.py
│   │   └── train_for_an_epoch.py
│   └── infer/
│       └── generate_failure_images.py
├── utils/
│   └── convert_dataset_to_yolo_format.py
├── best.pt                   # Trained YOLO26m weights (5 epochs)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Download the Dataset

Download BDD100K from [Google Drive](https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view) and extract it into a folder named `bdd100k_images_100k`.

After extraction, the directory structure should look like:

```
bdd100k_images_100k/
├── bdd100k_images_100k/bdd100k/images/100k/
│   ├── test/
│   ├── train/
│   └── val/
└── bdd100k_labels_release/bdd100k/labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

### 2. Build the Docker Image

```bash
docker build -t od-demo .
```

### 3. Run a Container

```bash
docker run -p 8501:8501 -it --name od_container \
  -v /path/to/bdd100k_images_100k:/app/bdd100k_images_100k \
  od-demo
```

If you face problems with mounting the dataset (I faced it since I was using a Fuse mount), you can
```bash
docker run -p 8501:8501 -d --name od_container od-demo
```

Then
```bash
docker cp path_to_dataset/. od_container:/app/bdd100k_images_100k/
```

Now, you can launch container using
```bash
docker exec -it od_container bash
```

Inside the container, the working directory is `/app/` and the layout mirrors the project structure above.

---

## Part 1: Data Analysis

### EDA

```bash
python data_analysis/EDA.py
```

This creates an `outputs/` directory containing JSON files with dataset statistics. These files are used by the dashboard.

### Dashboard

```bash
streamlit run data_analysis/dashboard.py
```

This launches an interactive dashboard in the browser (accessible at `http://localhost:8501`) with plots and insights from the analyzed data.

A detailed writeup is available in [`docs/data_analysis_report.md`](docs/data_analysis_report.md).

---

## Part 2: Model Training & Evaluation

### Background

No pre-trained model on BDD100K was available for out-of-the-box evaluation. A **YOLO26m** model was therefore trained for 5 epochs on the BDD100K training split. The trained model was then evaluated on the validation split. During training, the model was explicitly prevented from accessing the validation split (`val=False`).

Training was performed on a high-performance node with 8× NVIDIA A100 GPUs. YOLO26 is part of the [Ultralytics](https://github.com/ultralytics/ultralytics) library, which provides highly optimized pipelines for training and evaluation — hence the traditional dataloader → model → training loop approach was not adopted for the main experiment.

However, a **minimal training demo** that trains a dummy object detector on a small subset for 1 epoch is provided. It runs on CPU and requires no specialized hardware.

### Step 1: Convert Dataset to YOLO Format

```bash
python utils/convert_dataset_to_yolo_format.py
```

This converts the BDD100K annotations into YOLO format and creates a `bdd_yolo/` directory.

### Step 2: Dummy Training Demo (CPU)

```bash
python -m model.dummy_train.train_for_an_epoch
```

Trains a simple detector on a small subset for 1 epoch on CPU. No GPU required.

### Step 3: Evaluation

The trained weights (`best.pt`) are provided in the project root.

```bash
python model/infer/generate_failure_images.py
```

This performs a detailed evaluation, generating both quantitative metrics and qualitative visual assets into a `report_assets/` directory.

### Evaluation Report

For the full analysis — including per-class metrics, failure clustering, confusion matrices, and improvement suggestions — see [`docs/YOLO26m_Evaluation_Report.md`](docs/YOLO26m_Evaluation_Report.md).
