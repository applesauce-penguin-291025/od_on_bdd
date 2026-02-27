# YOLO26m Evaluation Report — BDD100K Object Detection

## 1. Executive Summary

This report presents a comprehensive evaluation of the **YOLO26m** (medium) model fine-tuned for 5 epochs on the **BDD100K** autonomous-driving object detection dataset. The model was evaluated on the full BDD100K validation split (10,000 images, 185,526 annotated instances across 10 classes). The evaluation covers quantitative metrics (mAP, precision, recall, F1), qualitative visual analysis (ground-truth vs. prediction comparisons), failure clustering, and actionable improvement suggestions.

**Overall Results (Validation Set):**

| Metric | Value |
|---|---|
| **mAP@50** | **0.5393** |
| **mAP@50-95** | **0.3064** |
| **Precision (mean)** | **0.7172** |
| **Recall (mean)** | **0.4867** |

---

## 2. Model Selection Rationale — Why YOLO26m?

**YOLO26** (Ultralytics, 2026) was chosen as the detection backbone for the following reasons:

| Factor | Rationale |
|---|---|
| **State-of-the-art architecture** | YOLO26 is the latest generation in the Ultralytics YOLO family, incorporating architectural improvements over YOLOv11/v12 including improved feature-pyramid design and attention mechanisms. It represents the current best trade-off between accuracy and speed for real-time detection. |
| **Medium variant ("m")** | The `m` (medium) variant was selected as a balanced choice — large enough to capture fine-grained distinctions between visually similar classes (e.g., bike vs. motor, person vs. rider) while remaining efficient enough for multi-GPU training on the full BDD100K dataset. It has 20.4M parameters and 67.9 GFLOPs — well-suited for 640×640 input on A100 GPUs. |
| **COCO-pretrained weights** | Starting from COCO-pretrained `yolo26m.pt` provides strong transfer learning — COCO contains many of the same object categories (person, car, truck, bus, bicycle, motorcycle, train, traffic light) as BDD100K, giving the model a significant head start. |
| **Autonomous-driving relevance** | Real-time inference speed (~1.5ms/image) makes YOLO26m deployable for on-vehicle perception, unlike heavier two-stage detectors. The BDD100K benchmark is specifically designed for autonomous-driving evaluation, making YOLO26 a natural fit. |
| **Ultralytics ecosystem** | The Ultralytics framework provides built-in validation, plotting (PR curves, confusion matrices), augmentation pipelines, and multi-GPU support — accelerating the experiment cycle. |

---

## 3. Model & Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO26m (20.4M params, 67.9 GFLOPs) |
| Pretrained | Yes (COCO pretrained weights — `yolo26m.pt`) |
| Dataset | BDD100K (full) — 10 classes |
| Training Epochs | 5 |
| Image Size | 640×640 |
| Batch Size | 128 |
| GPUs | 8× NVIDIA A100-SXM4-80GB |
| Optimizer | Auto (SGD) |
| LR₀ / LRf | 0.01 / 0.01 |
| Augmentations | Mosaic=1.0, Flipud=0.0, Fliplr=0.5, HSV, RandomErasing=0.4, RandAugment |
| AMP | Enabled |
| Input Resolution | 640×640 |
| **Validation during training** | **Disabled (`val=False`)** — see explanation below |

**Training Loss Curve (from `results.csv`):**

| Epoch | Box Loss | Cls Loss | DFL Loss | Time (s) |
|---|---|---|---|---|
| 1 | 1.9250 | 1.7436 | 0.00318 | 776 |
| 2 | 1.8861 | 1.1482 | 0.00308 | 1,475 |
| 3 | 1.8800 | 1.1081 | 0.00307 | 2,173 |
| 4 | 1.8334 | 1.0553 | 0.00294 | 2,923 |
| 5 | 1.8070 | 1.0247 | 0.00283 | 3,664 |

> **Observation:** All three losses are still decreasing monotonically at epoch 5 — the model is clearly **under-trained** and would benefit substantially from more epochs.

![Training Curves](report_assets/results.png)

### 3.1 Exact Commands Used

**Training command** (executed via CLI, commented in `train.py`):

```bash
yolo detect train \
  model=yolo26m.pt \
  data=/mnt/data_out/ALGO/soursar/od/bdd_yolo_full/bdd100k.yaml \
  epochs=5 \
  imgsz=640 \
  batch=128 \
  device=0,1,2,3,4,5,6,7 \
  workers=8 \
  amp=True \
  val=False
```

**Why `val=False`?** Validation was explicitly disabled during training to **prevent the model from seeing any validation data during the training loop**. This is critical for producing an honest, unbiased evaluation:
- With `val=True` (the default), YOLO runs a full validation pass after each epoch. While validation data is not used for gradient updates, the `best.pt` checkpoint is selected based on validation mAP — meaning the checkpoint itself is indirectly optimized on val data.
- By setting `val=False`, the model trains purely on the training split. The `best.pt` checkpoint saved is simply the final epoch's weights (epoch 5), with no val-based selection bias.
- Validation is then run **separately after training** using the script below, ensuring a clean train/val separation.

**Validation command** (executed via `train.py` Python API):

```python
from ultralytics import YOLO

model = YOLO("/mnt/data_out/ALGO/soursar/od/output/runs/detect/train2/weights/best.pt")
metrics = model.val(
    data="/mnt/data_out/ALGO/soursar/od/bdd_yolo_full/bdd100k.yaml",
    device=(0,1,2,3,4,5,6,7)
)

print(metrics.box.map50)     # mAP@50
print(metrics.box.map)       # mAP@50-95
print(metrics.box.maps)      # per-class AP
```

This loads the saved `best.pt` weights and evaluates on the full BDD100K validation split (10,000 images) — generating all quantitative metrics and visualization plots used in this report.

---

## 4. Quantitative Evaluation — Metrics Selection & Rationale

### 4.1 Why These Metrics?

| Metric | Why It Was Chosen |
|---|---|
| **mAP@50** | The standard PASCAL VOC metric. Measures detection quality at a lenient IoU threshold — captures whether the model finds objects "roughly in the right place." Critical for autonomous-driving where missing an object is worse than imprecise localization. |
| **mAP@50-95** | The stricter COCO-style metric averaged across IoU thresholds 0.50–0.95 in steps of 0.05. Rewards tight bounding-box regression — important for downstream tasks (tracking, planning) that need precise spatial extent. |
| **Precision** | Fraction of detections that are correct. High precision = fewer false alarms. Essential in driving: false positives can cause unnecessary braking or lane changes. |
| **Recall** | Fraction of ground-truth objects that are detected. High recall = fewer missed objects. Critical safety metric: a missed pedestrian or vehicle is potentially catastrophic. |
| **F1 Score** | Harmonic mean of precision and recall. Gives a single balanced measure per class — useful for ranking class-level performance and identifying the weakest links. |
| **Confusion Matrix** | Reveals systematic inter-class confusions (e.g., bike↔motor, person↔rider) and background false positives/negatives. Directly informs whether errors are detection failures or classification failures. |
| **Per-class box size analysis** | Correlates model performance with object scale — critical because small objects (traffic lights, distant pedestrians) are notoriously harder to detect. |

### 4.2 Per-Class Quantitative Results

| Class | Instances | % of Data | AP@50 | AP@50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **car** | 102,506 | 55.3% | **0.7980** | **0.4947** | 0.7946 | **0.7183** | **0.7545** |
| **traffic sign** | 34,908 | 18.8% | 0.6961 | 0.3703 | 0.6725 | 0.6642 | 0.6683 |
| **traffic light** | 26,885 | 14.5% | 0.6641 | 0.2586 | 0.6532 | 0.6561 | 0.6546 |
| **person** | 13,262 | 7.1% | 0.6517 | 0.3348 | 0.7321 | 0.5728 | 0.6428 |
| **truck** | 4,245 | 2.3% | 0.6373 | 0.4607 | 0.6515 | 0.5932 | 0.6210 |
| **bus** | 1,597 | 0.9% | 0.6129 | 0.4691 | 0.6753 | 0.5373 | 0.5984 |
| **bike** | 1,007 | 0.5% | 0.4753 | 0.2424 | 0.6673 | 0.3983 | 0.4988 |
| **rider** | 649 | 0.3% | 0.4451 | 0.2300 | 0.6702 | 0.4068 | 0.5063 |
| **motor** | 452 | 0.2% | 0.4095 | 0.2007 | 0.6557 | 0.3203 | 0.4303 |
| **train** | 15 | 0.0% | **0.0027** | **0.0024** | 1.0000 | **0.0000** | **0.0000** |

### 4.3 Key Findings from Quantitative Analysis

1. **Strong positive correlation between instance count and AP.** The top-3 classes by AP@50 (car, traffic sign, traffic light) are also the top-3 by instance count. This confirms that **data volume is the dominant driver of per-class performance**.

2. **The `train` class is effectively undetectable** (AP@50 = 0.003, recall = 0.0). With only 15 instances in 14 images, the model never learned this class. Precision = 1.0 is vacuous (it simply never predicts "train").

3. **Low-frequency classes (`rider`, `motor`, `bike`) have AP@50 in the 0.41–0.48 range** — roughly 40–50% lower than `car`. Their recall is particularly poor (0.32–0.41), suggesting the model frequently misses these small, rare objects.

4. **Precision is consistently higher than recall across all classes** (mean P=0.717 vs. mean R=0.487). The model is "conservative" — it generates fewer false positives but misses many true objects. This is characteristic of **under-training** (only 5 epochs).

5. **mAP@50-95 is substantially lower than mAP@50** (0.306 vs. 0.539), indicating that while the model finds objects at a coarse level, its **bounding-box regression is not yet precise** — again consistent with insufficient training.

---

## 5. Qualitative Evaluation — Visual Analysis

### 5.1 Visualization Tools Used

| Tool | Purpose |
|---|---|
| **YOLO built-in `val_batchN_labels.jpg` / `val_batchN_pred.jpg`** | Side-by-side ground truth vs. prediction on sample validation batches. Allows immediate visual inspection of what the model detects vs. misses. |
| **Confusion Matrix (`confusion_matrix.png`, `confusion_matrix_normalized.png`)** | Reveals systematic classification errors and background confusion patterns. |
| **PR / P / R / F1 Curve Plots** | Show how precision, recall, and F1 vary with confidence threshold — useful for choosing operating points. |
| **Label Distribution Plot (`labels.jpg`)** | Visualizes dataset-level class frequency and bounding-box spatial distribution — connects data properties to model behavior. |

### 5.2 Ground Truth vs. Predictions (Qualitative Samples)

The following image pairs show ground-truth annotations (left) and model predictions (right) from three validation batches:

| Ground Truth | Predictions |
|---|---|
| ![Batch 0 Labels](report_assets/val_batch0_labels.jpg) | ![Batch 0 Preds](report_assets/val_batch0_pred.jpg) |
| ![Batch 1 Labels](report_assets/val_batch1_labels.jpg) | ![Batch 1 Preds](report_assets/val_batch1_pred.jpg) |
| ![Batch 2 Labels](report_assets/val_batch2_labels.jpg) | ![Batch 2 Preds](report_assets/val_batch2_pred.jpg) |

**Visual observations:**
- **Cars in the foreground/midground** are consistently detected with high confidence. Bounding boxes are tight.
- **Small/distant pedestrians and cyclists** are frequently missed in predictions (visible in GT but absent in predictions).
- **Traffic lights** — especially small ones at a distance — are partially detected. Many small lights in crowded intersections are missed.
- **Nighttime/low-light images** show noticeably fewer predictions than daytime images, despite ground truth annotations being present.
- **Occluded vehicles** (partially behind other cars/objects) are sometimes missed or have poorly-fitted bounding boxes.


### 5.3 Curated Success & Failure Visualizations

Below are curated side-by-side images (LEFT = Ground Truth in **green**, RIGHT = Predictions in **red**) for specific success and failure scenarios. These were generated by running inference on targeted subsets of the validation set.

#### Where the Model Works Well — Car Detection

The `car` class is the model's strongest category (AP@50 = 0.798). Most cars — especially in the foreground and midground — are detected with tight bounding boxes:

![Success Car 0](report_assets/failure_analysis/success_car_0.jpg)
![Success Car 1](report_assets/failure_analysis/success_car_1.jpg)
![Success Car 2](report_assets/failure_analysis/success_car_2.jpg)

#### Where the Model Fails — Train Class (0% Recall)

The `train` class has **zero successful detections** across all 15 ground-truth instances (14 images). The model simply never predicts "train" — every GT train box is completely missed. Note the green GT boxes on the left that have no corresponding red prediction boxes on the right:

![Failure Train 0](report_assets/failure_analysis/failure_train_0.jpg)
![Failure Train 1](report_assets/failure_analysis/failure_train_1.jpg)
![Failure Train 2](report_assets/failure_analysis/failure_train_2.jpg)

> **Key observation:** The train objects are actually **large** in the images (avg. relative area = 0.032), so the failure is purely due to having only 15 training examples — far below the minimum needed for the model to learn this class.

#### Where the Model Fails — Small Traffic Lights

Traffic lights are 86% small objects. In crowded intersection scenes with many tiny lights, the model misses several of them:

![Failure Small TL 0](report_assets/failure_analysis/failure_small_tl_0.jpg)
![Failure Small TL 1](report_assets/failure_analysis/failure_small_tl_1.jpg)
![Failure Small TL 2](report_assets/failure_analysis/failure_small_tl_2.jpg)

#### Where the Model Fails — Rare Classes (Rider, Motor, Bike)

These three classes have low instance counts and the model frequently misses them, especially at smaller scales:

**Rider** (649 instances, AP@50 = 0.445):
![Failure Rider 0](report_assets/failure_analysis/failure_rider_0.jpg)
![Failure Rider 1](report_assets/failure_analysis/failure_rider_1.jpg)

**Motor** (452 instances, AP@50 = 0.410):
![Failure Motor 0](report_assets/failure_analysis/failure_motor_0.jpg)
![Failure Motor 1](report_assets/failure_analysis/failure_motor_1.jpg)

**Bike** (1,007 instances, AP@50 = 0.475):
![Failure Bike 0](report_assets/failure_analysis/failure_bike_0.jpg)
![Failure Bike 1](report_assets/failure_analysis/failure_bike_1.jpg)

#### Where the Model Fails — Night / Low-Light Scenes

In dark scenes (mean pixel brightness < 60), the model's detection count drops noticeably vs. ground truth:

![Failure Night 0](report_assets/failure_analysis/failure_night_0.jpg)
![Failure Night 1](report_assets/failure_analysis/failure_night_1.jpg)
![Failure Night 2](report_assets/failure_analysis/failure_night_2.jpg)


### 5.4 Confusion Matrix Analysis

![Confusion Matrix](report_assets/confusion_matrix.png)
![Normalized Confusion Matrix](report_assets/confusion_matrix_normalized.png)

**Key inter-class confusions observed:**
- **person ↔ rider**: Some rider instances get classified as persons (both are human-shaped; rider is distinguished by being on a bike/motorcycle).
- **bike ↔ motor**: Bicycle and motorcycle are visually similar at small scales, leading to mutual confusion.
- **Background FN (missed detections)**: The dominant error mode across all classes. The rightmost column / bottom row of the confusion matrix shows a large fraction of ground-truth objects classified as "background" (i.e., not detected at all). This reinforces the quantitative finding that **recall is the primary weakness**.

### 5.5 Precision-Recall and F1 Curves

| Curve | Plot |
|---|---|
| **Precision–Recall** | ![PR Curve](report_assets/BoxPR_curve.png) |
| **Precision vs. Confidence** | ![P Curve](report_assets/BoxP_curve.png) |
| **Recall vs. Confidence** | ![R Curve](report_assets/BoxR_curve.png) |
| **F1 vs. Confidence** | ![F1 Curve](report_assets/BoxF1_curve.png) |

**Observations from PR/F1 curves:**
- The **PR curve for `car`** has the largest area under the curve, confirming it as the best-performing class.
- The **`train` class PR curve** is essentially at zero — a flat line.
- The **F1 curve peak** is at a relatively low confidence threshold, suggesting the model is not producing high-confidence predictions for many classes. More training would shift these curves upward and to the right.
- The gap between precision and recall at the optimal F1 operating point confirms the model's conservative detection behavior.

---

## 6. Failure Clustering — Where and Why the Model Fails

### 6.1 Failure Cluster 1: Rare / Under-Represented Classes

| Class | Val Instances | AP@50 | Issue |
|---|---|---|---|
| train | 15 | 0.003 | Virtually zero data — unlearnable |
| motor | 452 | 0.410 | Very few examples, small objects |
| rider | 649 | 0.445 | Rare + visually similar to person |
| bike | 1,007 | 0.475 | Rare + confused with motor |

**Root Cause:** Severe class imbalance. `car` alone is 55.3% of all instances. The bottom 4 classes combined are only 1.1%. Standard loss functions weight all instances equally, so the gradient is dominated by frequent classes.

### 6.2 Failure Cluster 2: Small Objects

| Class | % Small Boxes (<0.001 rel. area) | AP@50-95 |
|---|---|---|
| traffic light | **86.2%** | 0.259 |
| traffic sign | **72.1%** | 0.370 |
| person | **44.9%** | 0.335 |
| car | **41.5%** | 0.495 |

**Root Cause:** At 640×640 input resolution, a box with relative area < 0.001 occupies fewer than ~20×20 pixels. The YOLO26m backbone downsamples by 8×/16×/32× at its three detection heads, meaning the smallest objects are represented by only 1–3 feature-map cells. This fundamentally limits detection of small, distant objects.

**Evidence:** `traffic light` has 86% small boxes and despite being the 3rd most frequent class (26,885 instances), its AP@50-95 is only 0.259 — significantly below `car` (0.495) which has far fewer small boxes proportionally.

### 6.3 Failure Cluster 3: Under-Training

| Evidence | Detail |
|---|---|
| Training losses still decreasing | Box loss: 1.925 → 1.807 (epoch 1→5); Cls loss: 1.744 → 1.025 |
| Val metrics only computed at epoch 5 | val=False was set during training; no early stopping possible |
| mAP@50 gap from SOTA | BDD100K SOTA is ~0.60+ mAP@50; our 0.539 has clear headroom |
| Recall << Precision | Model hasn't converged to confident detection — still conservative |

### 6.4 Failure Cluster 4: Challenging Conditions

From qualitative analysis of prediction images:
- **Night/Low-light scenes**: Noticeably fewer detections despite annotations being present.
- **Heavy occlusion**: Partially visible objects (especially in dense traffic) are frequently missed.
- **Dense object scenes**: When many small objects cluster together (e.g., a row of traffic lights at an intersection), the model misses several of them — likely NMS suppression or feature-level interference.

---

## 7. Data Analysis — Connecting Data to Performance

### 7.1 Validation Set Distribution

| Class | Instances | % of Data | Present in N Images | Avg. Relative Box Area |
|---|---|---|---|---|
| car | 102,506 | 55.3% | 9,879 | 0.01019 |
| traffic sign | 34,908 | 18.8% | 8,221 | 0.00130 |
| traffic light | 26,885 | 14.5% | 5,653 | 0.00054 |
| person | 13,262 | 7.1% | 3,220 | 0.00312 |
| truck | 4,245 | 2.3% | 2,689 | 0.02976 |
| bus | 1,597 | 0.9% | 1,242 | 0.03654 |
| bike | 1,007 | 0.5% | 578 | 0.00587 |
| rider | 649 | 0.3% | 515 | 0.00648 |
| motor | 452 | 0.2% | 334 | 0.00846 |
| train | 15 | 0.0% | 14 | 0.03189 |

### 7.2 Box Size Distribution

| Class | Small (<0.001) | Medium (0.001–0.01) | Large (≥0.01) |
|---|---|---|---|
| person | 44.9% | 49.4% | 5.7% |
| rider | 35.1% | 49.5% | 15.4% |
| car | 41.5% | 39.9% | 18.6% |
| truck | 16.1% | 44.6% | 39.2% |
| bus | 14.7% | 43.2% | 42.1% |
| train | 6.7% | 33.3% | 60.0% |
| motor | 27.9% | 51.3% | 20.8% |
| bike | 22.3% | 62.9% | 14.8% |
| traffic light | **86.2%** | 13.8% | 0.0% |
| traffic sign | **72.1%** | 26.2% | 1.6% |

### 7.3 Data-to-Performance Correlation

**Key pattern:** There is a clear correlation triangle: **Instance Count × Object Size → AP**.

- `car` wins on both axes: high count (55.3%) + moderate-to-large size → highest AP.
- `traffic light` has high count (14.5%) but is overwhelmingly small (86.2% small) → AP drops to 0.259 (AP@50-95).
- `bus`/`truck` have low count but are large objects → still achieve reasonable AP@50-95 (0.47/0.46), sometimes rivaling higher-count classes.
- `train` fails on count (only 15 instances) despite being large — confirming that **a minimum data threshold** is required regardless of object size.

![Label Distribution](report_assets/labels.jpg)

---

## 8. Suggested Improvements

### 8.1 Training Strategy Improvements

| Improvement | Expected Impact | Effort |
|---|---|---|
| **Train for more epochs (25–50+)** | High — losses are still decreasing; recall will improve significantly. mAP@50 likely to reach 0.58+ | Low |
| **Enable validation during training (`val=True`)** | Medium — enables early stopping, learning rate scheduling based on val metrics, and better checkpoint selection. Currently `best.pt` was only evaluated once at epoch 5 | Low |
| **Use cosine LR schedule (`cos_lr=True`)** | Medium — smoother convergence than the current step schedule | Low |
| **Increase image resolution (1280×1280)** | High for small objects — traffic lights (86% small) and traffic signs (72% small) would dramatically benefit. Doubles compute cost | Medium |

### 8.2 Data & Sampling Improvements

| Improvement | Expected Impact | Effort |
|---|---|---|
| **Class-weighted loss / focal loss** | Medium — upweight rare classes (rider, motor, bike, train) to combat the 55% car dominance | Low |
| **Oversample rare classes** | Medium — repeat images containing rare classes to balance the effective distribution | Low |
| **Add external data for `train` class** | High for this class — 15 instances is unlearnable. Source from COCO (`train` class) or other datasets | Medium |
| **Targeted augmentation for small objects** | Medium — copy-paste augmentation (`copy_paste > 0`) for small traffic lights/signs | Low |
| **Add night-specific augmentation** | Medium — heavier brightness/contrast jitter to improve robustness in low-light | Low |

### 8.3 Model Architecture Improvements

| Improvement | Expected Impact | Effort |
|---|---|---|
| **Use YOLO26l or YOLO26x** | Medium — larger backbone = better feature representation, especially for fine-grained classes | Low |
| **Multi-scale training (`multi_scale > 0`)** | Medium — exposes the model to varying object scales during training | Low |
| **Enable `mixup` augmentation** | Low-Medium — helps with generalization, especially for under-represented classes | Low |

### 8.4 Priority Ranking

Based on the failure analysis, the **top-3 highest-impact, lowest-effort improvements** are:

1. **Train for 25+ more epochs** — the single most impactful change; the model is clearly under-converged.
2. **Increase input resolution to 1280** — directly addresses the dominant failure mode (small objects = 86% of traffic lights, 72% of traffic signs).
3. **Enable class-weighted loss + copy-paste augmentation** — addresses the class imbalance problem that starves rare classes of gradient signal.

---

## 9. Summary

The YOLO26m model achieves a respectable baseline of **mAP@50 = 0.539** on BDD100K after only 5 epochs of training. However, the model is significantly under-trained (losses still decreasing, recall << precision), and performance on rare/small-object classes is poor. The three primary failure modes are:

1. **Under-training** (only 5 epochs; all losses monotonically decreasing)
2. **Small object detection** (traffic lights are 86% small, yet only 0.259 AP@50-95)
3. **Class imbalance** (`car` = 55% of data; bottom 4 classes = 1.1%)

Extending training to 25+ epochs, increasing input resolution, and addressing class imbalance through loss weighting and augmentation are expected to bring mAP@50 above 0.60 and substantially improve recall for safety-critical classes like pedestrians, cyclists, and motorcycles.

---

*Report generated from validation run on BDD100K val split (10,000 images). All plots are in the `report_assets/` directory.*
