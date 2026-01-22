# ===============================
# Crop Disease Prediction - Single File Code
# ===============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Configuration
# -------------------------------
DATASET_PATH = "dataset"   # dataset/train, dataset/val, dataset/test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------------
# 2. Data Generators
# -------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_test_gen.flow_from_directory(
    f"{DATASET_PATH}/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = val_test_gen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# 3. Model (Transfer Learning)
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# 4. Compile Model
# -------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 5. Train Model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
test_loss, test_accuracy = model.evaluate(test_data)
print("\nTest Accuracy:", test_accuracy)

# -------------------------------
# 7. Classification Report
# -------------------------------
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_data.class_indices.keys())
))

# -------------------------------
# 8. Save Model
# -------------------------------
model.save("crop_disease_model.h5")
print("\nModel saved as crop_disease_model.h5")
