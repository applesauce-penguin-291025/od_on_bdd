import json
import os
import shutil 
import random
from tqdm import tqdm


def create_bdd_yolo_dir_structure(base_dir='bdd_yolo'):
    """
    Create directory structure for YOLO formatted BDD100K dataset.
    The structure will be:
    base_dir/
        train/
            images/
            labels/
        val/
            images/
            labels/
    """
    os.makedirs(base_dir, exist_ok=True)
    splits = ['train', 'val']
    
    for split in splits:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
        img_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    


def convert_bdd_to_yolo(json_path, split, base_dir= 'bdd_yolo', img_width=1280, img_height=720):
    """
    Convert BDD100K dataset annotations to YOLO format.
    Args:
        json_path (str): Path to the BDD100K JSON annotation file.
        split (str): Dataset split ('train' or 'val').
        base_dir (str): Base directory for the YOLO formatted dataset (default: 'bdd_yolo').
        img_width (int): Width of the images (default: 1280).
        img_height (int): Height of the images (default: 720).

    Note that while all the samples in the validation set are used, only 3500 samples from the training set are processed.
    This is because a small subset of training data (5%) will serve as the data for demonstrationg the training pipeline.
    """
    class_map = {
        "person": 0, "rider": 1, "car": 2, "truck": 3, 
        "bus": 4, "train": 5, "motor": 6, "bike": 7, 
        "traffic light": 8, "traffic sign": 9
    }
    
    
    print("[INFO] Loading JSON annotations...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} annotations for {split} set.")

    if split == 'train' and len(data) > 3500:
        data = random.sample(data, 3500)

    for item in tqdm(data, desc=f"Processing {split} set"):
        img_name = item['name']
        shutil.copy(os.path.join('bdd100k_images_100k', 'bdd100k_images_100k', 'bdd100k', 'images', '100k',
                              split, img_name),
                    os.path.join(base_dir, split, 'images', img_name))
        label_file = os.path.join(base_dir, split, 'labels', img_name.replace('.jpg', '.txt'))
        
        with open(label_file, 'w') as f:
            for obj in item.get('labels', []):
                if 'box2d' not in obj: continue
                
                cls = obj['category']
                if cls not in class_map: continue
                
                box = obj['box2d']
                # Calculate YOLO normalized coordinates
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x = (box['x1'] + box['x2']) / 2.0
                y = (box['y1'] + box['y2']) / 2.0
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']
                
                f.write(f"{class_map[cls]} {x*dw:.6f} {y*dh:.6f} {w*dw:.6f} {h*dh:.6f}\n")


if __name__ == "__main__":
    base_dir = 'bdd_yolo'
    create_bdd_yolo_dir_structure(base_dir)
    convert_bdd_to_yolo('bdd100k_images_100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json',
                        'train',
                         base_dir)
    convert_bdd_to_yolo('bdd100k_images_100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                        'val',
                         base_dir)
    shutil.copy('bdd100k.yaml', os.path.join(base_dir, 'bdd100k.yaml'))