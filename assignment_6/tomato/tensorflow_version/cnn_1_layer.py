import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models

# ============================================================
# 1. SETTINGS
# ============================================================

# Detect dataset path
if os.path.exists("../train"):
    DATASET_PATH = "../train"
elif os.path.exists("train"):
    DATASET_PATH = "train"
elif os.path.exists("Sample_Tomato_Leaf_Disease_Dataset"):
    DATASET_PATH = "Sample_Tomato_Leaf_Disease_Dataset"
else:
    DATASET_PATH = "../train"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

print("TensorFlow Version:", tf.__version__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of Classes:", num_classes)


# ============================================================
# 3. NORMALIZATION
# ============================================================

normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# 4. CNN MODEL (ONLY 1 CONVOLUTION LAYER)
# ============================================================

model = models.Sequential([
    # CNN Layer
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    ),

    # Convert feature maps into 1D vector
    layers.Flatten(),

    # Output Layer
    layers.Dense(
        num_classes,
        activation='softmax'
    )
])


# ============================================================
# 5. COMPILE MODEL
# ============================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ============================================================
# 6. TRAIN MODEL
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

loss, accuracy = model.evaluate(val_ds)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy)


# ============================================================
# 8. ACCURACY AND LOSS GRAPHS
# ============================================================

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================

images, labels = next(iter(val_ds))

predictions = model.predict(images)

plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    plt.imshow(images[i])

    actual = class_names[labels[i]]
    predicted = class_names[np.argmax(predictions[i])]

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()
