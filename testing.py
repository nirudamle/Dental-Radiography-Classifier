import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict

# ── Set your dataset root path ──────────────────────────────────────────────
DATASET_PATH = r"C:\Users\nirud\internship\dental_radiography\dataset\Dental_Radiography"   # change to your actual path
SPLITS       = ["train", "test", "valid"]
CLASSES      = ["Cavity", "Implant", "Fillings", "Impacted Tooth", "Normal"]

# ── 1. Count images per class per split ─────────────────────────────────────
counts = defaultdict(dict)
for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        if os.path.exists(folder):
            counts[split][cls] = len(os.listdir(folder))
        else:
            counts[split][cls] = 0

print(f"{'Class':<20}", *[f"{s:>8}" for s in SPLITS])
print("-" * 50)
for cls in CLASSES:
    print(f"{cls:<20}", *[f"{counts[s][cls]:>8}" for s in SPLITS])

# ── 2. Bar chart of class distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, split in zip(axes, SPLITS):
    vals   = [counts[split][c] for c in CLASSES]
    colors = ["#1D9E75","#EF9F27","#7F77DD","#D85A30","#378ADD"]
    ax.bar(CLASSES, vals, color=colors)
    ax.set_title(f"{split} set distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

# ── 3. Show sample images from each class ───────────────────────────────────
fig, axes = plt.subplots(1, len(CLASSES), figsize=(18, 4))
for ax, cls in zip(axes, CLASSES):
    folder = os.path.join(DATASET_PATH, "train", cls)
    img_path = os.path.join(folder, os.listdir(folder)[0])
    img = mpimg.imread(img_path)
    ax.imshow(img, cmap="gray")
    ax.set_title(cls, fontsize=10)
    ax.axis("off")
plt.suptitle("Sample images — one per class", fontsize=13)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.show()