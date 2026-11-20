import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\nirud\internship\dental_radiography\dataset\Dental_Radiography"
MODELS_DIR   = r"C:\Users\nirud\internship\dental_radiography\models"
PLOTS_DIR    = r"C:\Users\nirud\internship\dental_radiography\plots"
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
NUM_CLASSES  = 5
CLASSES      = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Normal"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Generators ────────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1,
    horizontal_flip=True, fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=True
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "valid"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=False
)

# ── Softened class weights ────────────────────────────────────────────────────
raw_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
raw_weights = np.clip(raw_weights, 0.2, 5.0)
class_weight_dict = dict(enumerate(raw_weights))

print("Softened class weights:")
for cls, w in zip(CLASSES, raw_weights):
    print(f"  {cls:<20} → {w:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_mlp():
    # Flatten → Dense layers. No spatial awareness — baseline only.
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="MLP")
    return model

def build_cnn():
    # Conv blocks extract spatial features progressively
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        # Block 1 — low-level edges
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        # Block 2 — textures
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        # Block 3 — patterns
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        # Block 4 — high-level features
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="CNN")
    return model

def build_pretrained_model(base_model_fn, name):
    # Pretrained base — frozen initially, unfrozen in phase 2
    base = base_model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)   # uses IMG_SIZE
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))  # uses IMG_SIZE
    x = base(inputs, training=False)  # training=False keeps BN layers frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model, base

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(model_name, save_path):
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-8)
    ]

def plot_history(history_dict, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history_dict['accuracy'],     label='Train')
    ax1.plot(history_dict['val_accuracy'], label='Val')
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history_dict['loss'],     label='Train')
    ax2.plot(history_dict['val_loss'], label='Val')
    ax2.set_title(f"{model_name} — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_history.png"), dpi=150)
    plt.close()
    print(f"  Saved plot → {model_name}_history.png")

def evaluate_model(model, model_name):
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(f"\n{'='*55}\n  {model_name} — Classification Report\n{'='*55}")
    print(report)

    with open(os.path.join(PLOTS_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved plot → {model_name}_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_simple(model_name, model):
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    if os.path.exists(save_path):
        print(f"\n  Skipping {model_name} — already saved")
        return None

    print(f"\n{'█'*55}\n  Training: {model_name}\n{'█'*55}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=20,
        steps_per_epoch=200,      # 200 of 786 batches — 3x faster per epoch
        validation_steps=50,      # 50 of 88 val batches
        class_weight=class_weight_dict,
        callbacks=get_callbacks(model_name, save_path),
        verbose=1
    )
    plot_history(history.history, model_name)
    evaluate_model(model, model_name)
    results[model_name] = round(max(history.history['val_accuracy']) * 100, 2)
    print(f"  Best val accuracy: {results[model_name]}%")


def train_pretrained(model_name, base_model_fn):
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    if os.path.exists(save_path):
        print(f"\n  Skipping {model_name} — already saved")
        return None

    print(f"\n{'█'*55}\n  Training: {model_name}\n{'█'*55}")
    model, base = build_pretrained_model(base_model_fn, model_name)

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    # High LR is fine here — only the new Dense layers are updating
    print("\n  Phase 1 — training head only (5 epochs)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=5,
        steps_per_epoch=200,
        validation_steps=50,
        class_weight=class_weight_dict,
        verbose=1
    )

    # ── Phase 2: Unfreeze top 30 layers, fine-tune with tiny LR ─────────────
    # 1e-5 is critical — higher LR destroys ImageNet weights (catastrophic forgetting)
    print("\n  Phase 2 — fine-tuning top 30 layers (15 epochs)...")
    for layer in base.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=15,
        steps_per_epoch=200,
        validation_steps=50,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(model_name, save_path),
        verbose=1
    )

    # Combine phase 1 + phase 2 history for a single continuous plot
    combined = {
        'accuracy':     h1.history['accuracy']     + h2.history['accuracy'],
        'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy'],
        'loss':         h1.history['loss']         + h2.history['loss'],
        'val_loss':     h1.history['val_loss']     + h2.history['val_loss'],
    }
    plot_history(combined, model_name)
    evaluate_model(model, model_name)
    results[model_name] = round(max(combined['val_accuracy']) * 100, 2)
    print(f"  Best val accuracy: {results[model_name]}%")

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
results = {}

train_pretrained("EfficientNetB0", EfficientNetB0)  # fastest pretrained, run first
train_pretrained("MobileNetV2",    MobileNetV2)
train_simple("CNN",                build_cnn())
train_pretrained("ResNet50",       ResNet50)
train_pretrained("VGG16",          VGG16)
train_simple("MLP",                build_mlp())      # will skip if already saved

# ── Final comparison table ────────────────────────────────────────────────────
print(f"\n{'═'*45}")
print("  MODEL COMPARISON (Best Val Accuracy)")
print('═'*45)
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, acc) in enumerate(sorted_results, 1):
    bar = '█' * int(acc / 2)
    print(f"  {rank}. {name:<18} {acc:>6}%  {bar}")

best_model_name = sorted_results[0][0]
print(f"\n  Best model : {best_model_name}")
print(f"  Saved to   : {MODELS_DIR}\\{best_model_name}_best.h5")