# ===============================
# Predict Crop Disease - PlantVillage Dataset
# ===============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -------------------------------
# 1. Load trained model
# -------------------------------
MODEL_PATH = "crop_disease_model.h5"  # or .keras
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# 2. Image size
# -------------------------------
IMG_SIZE = (224, 224)

# -------------------------------
# 3. Class mapping
# -------------------------------
class_names = [
    "Pepper__bell__Bacterial_spot",
    "Pepper__bell__healthy",
    "Potato__Early_blight",
    "Potato__healthy",
    "Potato__Late_blight",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__healthy",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite"
]

# -------------------------------
# 4. Load and preprocess image
# -------------------------------
IMAGE_PATH = "image2.webp"  # replace with your test image path

img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -------------------------------
# 5. Predict
# -------------------------------
pred = model.predict(img_array)
predicted_index = np.argmax(pred)
predicted_class = class_names[predicted_index]

print("Predicted Class Index:", predicted_index)
print("Predicted Disease:", predicted_class)
