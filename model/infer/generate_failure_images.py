"""
Generate curated success/failure visualization images for the evaluation report.
"""
import os, random, cv2, numpy as np
from ultralytics import YOLO
from collections import defaultdict

random.seed(42)

MODEL_PATH = "best.pt"
DATA_ROOT  = "bdd_yolo"
IMG_DIR    = os.path.join(DATA_ROOT, "val/images")
LBL_DIR    = os.path.join(DATA_ROOT, "val/labels")
OUT_DIR    = "report_assets/failure_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = {0:'person', 1:'rider', 2:'car', 3:'truck', 4:'bus',
         5:'train', 6:'motor', 7:'bike', 8:'traffic light', 9:'traffic sign'}
COLORS_GT   = (0, 255, 0)
COLORS_PRED = (0, 0, 255)

model = YOLO(MODEL_PATH)

def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def draw_boxes(img, boxes, color, label_prefix="", with_names=True):
    for item in boxes:
        if len(item) == 5:
            cls, x1, y1, x2, y2 = item
            conf = None
        else:
            cls, x1, y1, x2, y2, conf = item
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if with_names:
            name = NAMES.get(cls, str(cls))
            txt = f"{label_prefix}{name}"
            if conf is not None:
                txt += f" {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def make_side_by_side(img_path, label_path, title=""):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gt_boxes = load_gt(label_path, w, h)
    img_gt = img.copy()
    draw_boxes(img_gt, gt_boxes, COLORS_GT, label_prefix="GT:")
    results = model.predict(img_path, imgsz=640, conf=0.25, device=0, verbose=False)
    pred_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            pred_boxes.append((cls, x1, y1, x2, y2, conf))
    img_pred = img.copy()
    draw_boxes(img_pred, pred_boxes, COLORS_PRED)
    combined = np.hstack([img_gt, img_pred])
    bar_h = 40
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{title}  |  LEFT: Ground Truth (green)    RIGHT: Predictions (red)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    combined = np.vstack([bar, combined])
    return combined, len(gt_boxes), len(pred_boxes)

# Index images by class
print("Indexing labels by class...")
class_to_images = defaultdict(list)
all_label_files = [f for f in os.listdir(LBL_DIR) if f.endswith('.txt')]
for lf in all_label_files:
    with open(os.path.join(LBL_DIR, lf)) as f:
        classes_in_file = set()
        for line in f:
            cls = int(line.strip().split()[0])
            classes_in_file.add(cls)
        for cls in classes_in_file:
            class_to_images[cls].append(lf.replace('.txt', ''))
print(f"Indexed {len(all_label_files)} label files.")

# === SUCCESS: Car detection ===
print("\n=== SUCCESS cases (car detection) ===")
car_images = class_to_images[2]
random.shuffle(car_images)
count = 0
for img_name in car_images[:50]:
    img_path = os.path.join(IMG_DIR, img_name + ".jpg")
    lbl_path = os.path.join(LBL_DIR, img_name + ".txt")
    if not os.path.exists(img_path): continue
    result = make_side_by_side(img_path, lbl_path, title=f"SUCCESS: Car Detection - {img_name}")
    if result is None: continue
    combined, n_gt, n_pred = result
    if n_gt >= 3 and n_pred >= max(1, int(n_gt * 0.6)):
        out_path = os.path.join(OUT_DIR, f"success_car_{count}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  Saved: {out_path} (GT={n_gt}, Pred={n_pred})")
        count += 1
        if count >= 3: break

# === FAILURE: Train class ===
print("\n=== FAILURE cases (train class) ===")
train_images = class_to_images[5]
count = 0
for img_name in train_images:
    img_path = os.path.join(IMG_DIR, img_name + ".jpg")
    lbl_path = os.path.join(LBL_DIR, img_name + ".txt")
    if not os.path.exists(img_path): continue
    result = make_side_by_side(img_path, lbl_path, title=f"FAILURE: Train Class (0% recall) - {img_name}")
    if result is None: continue
    combined, n_gt, n_pred = result
    out_path = os.path.join(OUT_DIR, f"failure_train_{count}.jpg")
    cv2.imwrite(out_path, combined)
    print(f"  Saved: {out_path} (GT={n_gt}, Pred={n_pred})")
    count += 1
    if count >= 3: break

# === FAILURE: Small traffic lights ===
print("\n=== FAILURE cases (small traffic lights) ===")
tl_images = class_to_images[8]
random.shuffle(tl_images)
count = 0
for img_name in tl_images[:100]:
    img_path = os.path.join(IMG_DIR, img_name + ".jpg")
    lbl_path = os.path.join(LBL_DIR, img_name + ".txt")
    if not os.path.exists(img_path): continue
    gt = load_gt(lbl_path, 1280, 720)
    tl_gt = [b for b in gt if b[0] == 8]
    small_tl = [b for b in tl_gt if (b[3]-b[1])*(b[4]-b[2]) < 400]
    if len(small_tl) < 4: continue
    result = make_side_by_side(img_path, lbl_path, title=f"FAILURE: Small Traffic Lights - {img_name}")
    if result is None: continue
    combined, n_gt, n_pred = result
    out_path = os.path.join(OUT_DIR, f"failure_small_tl_{count}.jpg")
    cv2.imwrite(out_path, combined)
    print(f"  Saved: {out_path} (GT={n_gt}, Pred={n_pred}, small_tl={len(small_tl)})")
    count += 1
    if count >= 3: break

# === FAILURE: Rare classes ===
print("\n=== FAILURE cases (rare classes) ===")
for cls_id, cls_name in [(1, 'rider'), (6, 'motor'), (7, 'bike')]:
    imgs = class_to_images[cls_id]
    random.shuffle(imgs)
    count = 0
    for img_name in imgs[:50]:
        img_path = os.path.join(IMG_DIR, img_name + ".jpg")
        lbl_path = os.path.join(LBL_DIR, img_name + ".txt")
        if not os.path.exists(img_path): continue
        gt = load_gt(lbl_path, 1280, 720)
        cls_gt_count = sum(1 for b in gt if b[0] == cls_id)
        if cls_gt_count < 2: continue
        result = make_side_by_side(img_path, lbl_path, title=f"FAILURE: {cls_name} - {img_name}")
        if result is None: continue
        combined, n_gt, n_pred = result
        out_path = os.path.join(OUT_DIR, f"failure_{cls_name}_{count}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  Saved: {out_path} (GT={n_gt}, Pred={n_pred})")
        count += 1
        if count >= 2: break

# === FAILURE: Night scenes ===
print("\n=== FAILURE cases (night/dark) ===")
random.shuffle(all_label_files)
count = 0
for lf in all_label_files[:500]:
    img_name = lf.replace('.txt', '')
    img_path = os.path.join(IMG_DIR, img_name + ".jpg")
    if not os.path.exists(img_path): continue
    img = cv2.imread(img_path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness > 60: continue
    lbl_path = os.path.join(LBL_DIR, lf)
    gt = load_gt(lbl_path, img.shape[1], img.shape[0])
    if len(gt) < 5: continue
    result = make_side_by_side(img_path, lbl_path, title=f"FAILURE: Night (brightness={mean_brightness:.0f}) - {img_name}")
    if result is None: continue
    combined, n_gt, n_pred = result
    out_path = os.path.join(OUT_DIR, f"failure_night_{count}.jpg")
    cv2.imwrite(out_path, combined)
    print(f"  Saved: {out_path} (GT={n_gt}, Pred={n_pred}, brightness={mean_brightness:.0f})")
    count += 1
    if count >= 3: break

print("\n=== Done! ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
