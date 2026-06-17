import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration and Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed_augmented"  # Switch to processed_augmented if testing noise
LABELS_CSV = DATA_DIR / "labels_augmented.csv"  # Switch to labels_augmented.csv if testing noise

IMG_WIDTH = 256
IMG_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS = 30  # EarlyStopping will stop it when needed

# Map string labels to numeric integers
LANG_MAP = {'pl': 0, 'nl': 1, 'pt': 2}

print("Initializing regularized training script...")

# Image pipeline function with normalization
def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img, label

# Data Loading
if not LABELS_CSV.exists():
    raise FileNotFoundError(f"Missing file: {LABELS_CSV}. Run src/dataset.py first.")

df = pd.read_csv(LABELS_CSV)

file_paths = [str(PROCESSED_DIR / row['filename']) for _, row in df.iterrows()]
labels = [LANG_MAP[row['language']] for _, row in df.iterrows()]

# Train-test split (80% train, 20% validation)
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=27, stratify=labels
)

print(f"Dataset loaded. Train samples: {len(X_train_paths)}, Test samples: {len(X_test_paths)}")

# Create TensorFlow Dataset pipelines - with prefetch for performance
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
    .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(len(X_train_paths))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
    .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Regularized CNN Model Architecture
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense layers
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
MODEL_SAVE_PATH = Path("models")
MODEL_SAVE_PATH.mkdir(exist_ok=True)

callbacks = [
    # Stop training when val_loss stops improving for 5 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,  # Revert to best epoch automatically
        verbose=1
    ),
    # Save the best model during training
    ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH / "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Training
print("\nStarting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_SAVE_PATH / "language_recognition_model.keras")
print(f"\nTraining complete. Model saved to: models/language_recognition_model.keras")
print(f"Best model saved to: models/best_model.keras")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train accuracy')
ax1.plot(history.history['val_accuracy'], label='Val accuracy')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train loss')
ax2.plot(history.history['val_loss'], label='Val loss')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / "training_history.png", dpi=100)
plt.close()
print(f"Training plot saved to: models/training_history.png")

# Final evaluation
print("\n--- Final Evaluation on Test Set ---")
test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Test loss:     {test_loss:.4f}")