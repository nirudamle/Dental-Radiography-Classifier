import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\nirud\internship\dental_radiography\dataset\Dental_Radiography"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
CLASSES      = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Normal"]

# ── Train generator (with augmentation) ──────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalize pixel values 0-1
    rotation_range=15,         # random rotation ±15°
    width_shift_range=0.1,     # horizontal shift
    height_shift_range=0.1,    # vertical shift
    zoom_range=0.1,            # slight zoom
    horizontal_flip=True,      # flip (X-rays are symmetric)
    fill_mode='nearest'        # fill gaps after rotation/shift
)

# ── Val/Test generators (NO augmentation — only normalize) ────────────────────
val_test_datagen = ImageDataGenerator(rescale=1./255)

# ── Load data from folders ────────────────────────────────────────────────────
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # one-hot encoded labels
    classes=CLASSES,            # fix class order across all generators
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "valid"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False               # don't shuffle val — need consistent order for metrics
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

# ── Compute class weights ─────────────────────────────────────────────────────
# Gets the integer label for each training sample
train_labels = train_gen.classes  

class_weights = compute_class_weight(
    class_weight='balanced',    # sklearn formula: total / (n_classes * class_count)
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

print("Class indices:", train_gen.class_indices)
print("\nClass weights:")
for cls, w in zip(CLASSES, class_weights):
    print(f"  {cls:<20} → weight: {w:.4f}")

# ── Visualize augmented samples ───────────────────────────────────────────────
# Grab one batch and show 10 augmented versions of different images
images, labels = next(train_gen)
label_names    = list(train_gen.class_indices.keys())

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])          # already normalized 0-1, imshow handles it
    ax.set_title(label_names[np.argmax(labels[i])], fontsize=9)
    ax.axis("off")
plt.suptitle("Sample augmented training images", fontsize=13)
plt.tight_layout()
plt.savefig("augmented_samples.png", dpi=150)
plt.show()

print("\nGenerator summary:")
print(f"  Train batches : {len(train_gen)}")
print(f"  Val batches   : {len(val_gen)}")
print(f"  Test batches  : {len(test_gen)}")
print(f"  Image shape   : {images.shape[1:]}")