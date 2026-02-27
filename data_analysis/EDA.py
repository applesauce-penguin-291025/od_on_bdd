"""
BDD100K Object Detection Dataset Analysis

This module parses BDD100K detection labels and computes:
- Bounding box counts per class
- Image counts per class
- Time-of-day and weather distributions
- Bounding box area statistics and CDF

Author: <your name>
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

# Headless backend for Docker
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

SPLITS = ["train", "val"]

def get_labels_path(split: str) -> Path:
    return Path(
        f"bdd100k_images_100k/"
        f"bdd100k_labels_release/bdd100k/labels/"
        f"bdd100k_labels_images_{split}.json"
    )


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Parsing utilities
# -----------------------------

def load_labels(json_path: Path) -> List[Dict]:
    """
    Load BDD100K label JSON.

    Args:
        json_path: Path to labels JSON.

    Returns:
        List of per-image label dictionaries.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Label file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def extract_detection_labels(record: Dict) -> List[Dict]:
    """
    Extract detection labels (those with box2d).

    Args:
        record: Single image annotation record.

    Returns:
        List of detection label dictionaries.
    """
    return [
        label
        for label in record.get("labels", [])
        if "box2d" in label
    ]


# -----------------------------
# Statistics computation
# -----------------------------

def compute_class_statistics(label_infos: List[Dict]) -> Dict[str, Counter]:
    """
    Compute bbox-level and image-level class statistics.

    Returns:
        Dictionary with keys:
        - bbox_counter
        - image_counter
    """
    bbox_counter = Counter()
    image_counter = Counter()

    for record in label_infos:
        classes_in_image = set()

        for label in extract_detection_labels(record):
            cls = label["category"]
            bbox_counter[cls] += 1
            classes_in_image.add(cls)

        for cls in classes_in_image:
            image_counter[cls] += 1

    return {
        "bbox_counter": bbox_counter,
        "image_counter": image_counter,
    }


def compute_annotation_anomalies(label_infos):
    """
    Analyze annotation quality issues that can be detected
    without image resolution metadata.
    """
    stats = {
        "zero_area": 0,
        "negative_dims": 0,
        "tiny_boxes": Counter(),
        "extreme_aspect_ratio": Counter(),
        "empty_images": 0,
    }

    TINY_AREA_THRESH = 10.0          # pixels^2
    ASPECT_RATIO_THRESH = 10.0       # max(w/h, h/w)

    for record in label_infos:
        labels = [
            lbl for lbl in record.get("labels", [])
            if "box2d" in lbl
        ]

        if not labels:
            stats["empty_images"] += 1
            continue

        for lbl in labels:
            cls = lbl["category"]
            b = lbl["box2d"]

            w = float(b["x2"]) - float(b["x1"])
            h = float(b["y2"]) - float(b["y1"])

            if w <= 0 or h <= 0:
                stats["negative_dims"] += 1
                continue

            area = w * h
            aspect_ratio = max(w / h, h / w)

            if area == 0:
                stats["zero_area"] += 1

            if area < TINY_AREA_THRESH:
                stats["tiny_boxes"][cls] += 1

            if aspect_ratio > ASPECT_RATIO_THRESH:
                stats["extreme_aspect_ratio"][cls] += 1

    return stats


def print_annotation_anomalies(anomalies):
    print("\nAnnotation Quality Analysis")

    print(f"Images with no detection labels: {anomalies['empty_images']}")
    print(f"Boxes with non-positive dimensions: {anomalies['negative_dims']}")
    print(f"Boxes with zero area: {anomalies['zero_area']}")

    print("\nTiny bounding boxes (<10 px²) per class:")
    for cls, cnt in anomalies["tiny_boxes"].most_common():
        print(f"{cls:>15s}: {cnt}")

    print("\nExtreme aspect ratio boxes (>10:1) per class:")
    for cls, cnt in anomalies["extreme_aspect_ratio"].most_common():
        print(f"{cls:>15s}: {cnt}")


def compute_attribute_statistics(label_infos: List[Dict]) -> Dict[str, Counter]:
    """
    Compute time-of-day and weather distributions.

    Returns:
        Dictionary with keys:
        - time_of_day
        - weather
    """
    time_of_day = Counter()
    weather = Counter()
    scene = Counter()

    for record in label_infos:
        attrs = record.get("attributes", {})
        time_of_day[attrs.get("timeofday", "unknown")] += 1
        weather[attrs.get("weather", "unknown")] += 1
        scene[attrs.get("scene", "unknown")] += 1

    return {
        "time_of_day": time_of_day,
        "weather": weather,
        "scene": scene,
    }


def compute_bbox_areas(label_infos: List[Dict]) -> np.ndarray:
    """
    Compute bounding box areas in pixel space.

    Note:
        Image dimensions are NOT stored in labels JSON,
        so areas are absolute pixel areas, not normalized.

    Returns:
        Numpy array of bbox areas.
    """
    areas = []

    for record in label_infos:
        for label in extract_detection_labels(record):
            b = label["box2d"]

            w = max(0.0, float(b["x2"]) - float(b["x1"]))
            h = max(0.0, float(b["y2"]) - float(b["y1"]))
            area = w * h

            if area > 0:
                areas.append(area)

    return np.asarray(areas, dtype=np.float64)


# -----------------------------
# Visualization
# -----------------------------

def plot_bbox_area_cdf(areas: np.ndarray, split: str, out_dir: Path) -> None:
    """
    Plot CDF of bounding box areas.

    CDF definition:
        For sorted areas x,
        CDF(x[i]) = (i + 1) / N

    This answers:
        "What fraction of boxes are smaller than a given area?"

    Args:
        areas: Array of bbox areas.
        split: Dataset split name.
        out_dir: Directory to save plots.
    """
    if areas.size == 0:
        raise ValueError("No bounding box areas to plot.")

    x = np.sort(areas)
    cdf = np.arange(1, x.size + 1, dtype=np.float64) / x.size

    plt.figure(figsize=(6, 4))
    plt.plot(x, cdf, linewidth=2)
    plt.xlabel("Bounding box area (pixels²)")
    plt.xlim(0, 10000)
    plt.ylabel("CDF")
    plt.title(f"BBox Area CDF ({split})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"bbox_area_cdf_{split}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Saved bbox area CDF to {out_path}")


# -----------------------------
# Reporting helpers
# -----------------------------

def print_counter(title: str, counter: Counter) -> None:
    """Pretty-print a Counter."""
    print(f"\n{title}")
    for key, val in counter.most_common():
        print(f"{key:>15s}: {val}")


def print_area_quantiles(areas: np.ndarray) -> None:
    """Print useful area quantiles."""
    qs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qvals = np.quantile(areas, qs)

    print("\nBBox area quantiles (pixels²):")
    for q, v in zip(qs, qvals):
        print(f"{int(q * 100):>2d}th percentile: {v:.2f}")


def save_analysis_results(
    split: str,
    class_stats,
    attr_stats,
    anomalies,
    areas: np.ndarray,
    out_dir: Path,
) -> None:
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = dict(
        zip(
            [f"p{int(q * 100)}" for q in qs],
            np.quantile(areas, qs).tolist(),
        )
    )

    output = {
        "class_stats": {
            "bbox_counter": dict(class_stats["bbox_counter"]),
            "image_counter": dict(class_stats["image_counter"]),
        },
        "attribute_stats": {
            k: dict(v) for k, v in attr_stats.items()
        },
        "annotation_anomalies": {
            "tiny_boxes": dict(anomalies["tiny_boxes"]),
            "extreme_aspect_ratio": dict(anomalies["extreme_aspect_ratio"]),
            "empty_images": anomalies["empty_images"],
        },
        "bbox_area_quantiles": quantiles,
    }

    out_path = out_dir / f"analysis_{split}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Saved analysis results to {out_path}")

# -----------------------------
# Main entrypoint
# -----------------------------

def main() -> None:
    for split in SPLITS:
        labels_json = get_labels_path(split)
        print(f"\n{'=' * 60}")
        print(f"[INFO] Processing split: {split}")
        print(f"{'=' * 60}")
        print(f"[INFO] Loading labels from {labels_json}")
        label_infos = load_labels(labels_json)
        print(f"[INFO] Loaded {len(label_infos)} image records")

        class_stats = compute_class_statistics(label_infos)
        attr_stats = compute_attribute_statistics(label_infos)
        bbox_areas = compute_bbox_areas(label_infos)
        anomalies = compute_annotation_anomalies(label_infos)

        print_counter(
            "Detection class distribution (number of bboxes)",
            class_stats["bbox_counter"],
        )

        print_counter(
            "Detection class distribution (number of images)",
            class_stats["image_counter"],
        )

        print_counter(
            "Time-of-day distribution",
            attr_stats["time_of_day"],
        )

        print_counter(
            "Weather distribution",
            attr_stats["weather"],
        )

        print_counter(
            "Scene distribution",
            attr_stats["scene"],
        )

        print_area_quantiles(bbox_areas)
        # plot_bbox_area_cdf(bbox_areas, SPLIT, OUTPUT_DIR)

        print_annotation_anomalies(anomalies)
        save_analysis_results(
            split=split,
            class_stats=class_stats,
            attr_stats=attr_stats,
            anomalies=anomalies,
            areas=bbox_areas,
            out_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()