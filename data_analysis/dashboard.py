import json
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

STATS_DIR = Path("outputs")

st.set_page_config(layout="wide")
st.title("BDD100K Object Detection – Data Analysis Dashboard")


# -----------------------------
# Load precomputed stats ONLY
# -----------------------------

@st.cache_data
def load_stats(split: str) -> dict:
    path = STATS_DIR / f"analysis_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


split = st.sidebar.selectbox("Dataset split", ["train", "val"])
stats = load_stats(split)


# -----------------------------
# Class Distribution
# -----------------------------

st.header("Class Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bounding boxes per class")
    st.bar_chart(stats["class_stats"]["bbox_counter"])

with col2:
    st.subheader("Images per class")
    st.bar_chart(stats["class_stats"]["image_counter"])


# -----------------------------
# Scene Attributes
# -----------------------------

st.header("Scene Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Time of Day")
    st.bar_chart(stats["attribute_stats"]["time_of_day"])

with col2:
    st.subheader("Weather")
    st.bar_chart(stats["attribute_stats"]["weather"])

with col3:
    st.subheader("Scene Type")
    st.bar_chart(stats["attribute_stats"]["scene"])


# -----------------------------
# Object Scale (Quantile-based)
# -----------------------------

st.header("Object Scale Analysis")

quantiles = stats["bbox_area_quantiles"]
labels = list(quantiles.keys())
values = list(quantiles.values())

fig, ax = plt.subplots()
ax.plot(labels, values, marker="o")
ax.set_xlabel("Quantile")
ax.set_ylabel("Bounding box area (pixels²)")
ax.set_title("Bounding Box Area Quantiles")
ax.grid(alpha=0.3)

st.pyplot(fig)

st.markdown(
    f"""
**Interpretation:**
- 50% of bounding boxes are smaller than **{quantiles['p50']:.0f} px²**
- 90% are smaller than **{quantiles['p90']:.0f} px²**
- The long tail indicates presence of very large objects
"""
)


# -----------------------------
# Annotation Quality
# -----------------------------

st.header("Annotation Quality")

anomalies = stats["annotation_anomalies"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Images with no detection labels", anomalies["empty_images"])

with col2:
    st.metric(
        "Tiny boxes (<10 px²)",
        sum(anomalies["tiny_boxes"].values()),
    )

with col3:
    st.metric(
        "Extreme aspect ratio boxes",
        sum(anomalies["extreme_aspect_ratio"].values()),
    )

st.subheader("Tiny Bounding Boxes per Class")
st.bar_chart(anomalies["tiny_boxes"])

st.subheader("Extreme Aspect Ratio Boxes per Class")
st.bar_chart(anomalies["extreme_aspect_ratio"])