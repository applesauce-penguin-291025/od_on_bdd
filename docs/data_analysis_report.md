# BDD100K Object Detection – Data Analysis

## Objective
The goal of this analysis is to understand the statistical properties,
biases, and anomalies present in the BDD100K dataset for the task of
object detection. The analysis focuses exclusively on labels that
contain 2D bounding boxes.

---

## Dataset Overview
- Dataset: BDD100K
- Task: Object Detection
- Splits analyzed: Train and Validation
- Annotations: JSON-based, per-image metadata with bounding boxes

Note: Image resolution metadata is not present in the label files.
Bounding box areas are therefore computed in absolute pixel space.

---

## Class Distribution

### Bounding Box Level
We compute the total number of bounding boxes per class to assess
instance-level imbalance.

**Observation:**
- Certain classes (e.g., `car`) dominate the dataset.
- Long-tail behavior is evident for rare classes.

### Image Level
We also compute the number of images containing at least one instance
of each class.

**Observation:**
- Some classes appear frequently but with few instances per image,
  while others show clustering behavior.

This distinction is important for training stability and sampling
strategies.

---

## Contextual Bias Analysis

### Time of Day
We analyze the distribution of images across time-of-day categories.

**Observation:**
- The dataset is biased toward daytime driving scenes.
- Night-time scenes are underrepresented, which may affect
  generalization.

### Weather Conditions
Weather metadata reveals uneven representation across conditions.

**Observation:**
- Clear weather dominates the dataset.
- Adverse weather conditions are comparatively rare.

---

## Object Scale Analysis

Bounding box area distributions are analyzed using cumulative
distribution functions (CDFs).

The CDF answers the question:
> What fraction of objects are smaller than a given bounding box area?

**Key Findings:**
- A significant fraction of objects are small in pixel area.
- This effect is class-dependent, with traffic-related objects being
  particularly small.

This has implications for anchor design and feature pyramid selection.

---

## Anomalies and Patterns

We identify several noteworthy patterns:
- Extreme aspect ratios for certain object classes
- Very small bounding boxes that may be challenging to detect
- Contextual bias correlating object presence with time of day

These findings motivate targeted data augmentation and evaluation
strategies.

---

## Annotation Quality Analysis

In addition to distributional statistics, we analyze annotation quality
issues that can be detected directly from the label JSON files, without
requiring image resolution metadata.

---

### Images Without Detection Annotations

The label JSON file does not contain any image records without detection
labels. As a result, the analysis reports **zero images with empty
annotations** when operating solely on the annotation file.

However, when cross-referenced against the training image directory, we
observe that **137 training images do not have corresponding annotation
entries**. These images represent true background-only samples and are
not explicitly listed in the label JSON.

**Implication:**  
Such samples should be handled explicitly during training (e.g.,
negative sampling or exclusion), as they do not contribute positive
detection targets.

---

### Degenerate Bounding Boxes

No bounding boxes with zero or negative width/height were found in the
annotation files.

**Observation:**  
This indicates strong geometric consistency and a high level of basic
annotation quality for bounding box coordinates.

---

### Extremely Small Bounding Boxes

Bounding boxes with very small pixel areas (<10 pixels²) were observed,
with the following per-class distribution:

- `car`: 46
- `traffic light`: 18
- `traffic sign`: 14
- `person`: 8
- `truck`: 3
- `rider`: 1
- `bus`: 1
- `bike`: 1

**Observation:**  
Although rare in absolute terms, these very small objects are unlikely to
contain meaningful visual information and may introduce noise during
training, particularly for distant traffic-related objects.

---

### Extreme Aspect Ratio Bounding Boxes

A non-trivial number of bounding boxes exhibit extreme aspect ratios
greater than 10:1. The most affected classes include:

- `car`: 167
- `traffic sign`: 121
- `person`: 72
- `traffic light`: 31
- `truck`: 13
- `train`: 8
- `bus`: 4
- `bike`: 3
- `rider`: 2

**Observation:**  
These extreme shapes may arise from occlusion, partial visibility, or
annotation ambiguity. Such cases can negatively impact anchor-based
detectors and motivate the use of multi-scale or anchor-free detection
architectures.

---

### Summary

The BDD100K detection annotations exhibit strong geometric validity, with
no degenerate bounding boxes observed. However, the presence of extremely
small objects, extreme aspect ratios, and background-only images not
explicitly represented in the label JSON introduces challenges for
object detection models.

These findings highlight the need for:
- robustness to small object detection
- careful handling of background-only samples
- architectures that support diverse object scales and shapes

---

### Limitations

Bounding boxes extending beyond image boundaries cannot be reliably
detected from the label JSON alone, as image resolution metadata is not
included. Such validation would require loading the corresponding image
files.


## Dashboard
An interactive dashboard is provided to explore these statistics
visually. All dataset statistics are computed in a batch analysis 
step and stored as structured JSON artifacts. This avoids repeated
parsing of large annotation files and ensures that the interactive 
dashboard remains responsive. 
The dashboard can be launched using:

```bash
streamlit run data_analysis/dashboard.py


