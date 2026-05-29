import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
import os

MODEL_PATH = "small_inception_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Model file not found!")
    exit()

model = load_model(MODEL_PATH)
print("Model Loaded")

if len(sys.argv) < 2:
    print("Usage: python test_model.py <image_path>")
    exit()

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print("Image not found!")
    exit()

img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)

if pred[0][0] > 0.5:
    print("Prediction: Diseased")
else:
    print("Prediction: Healthy")

print("Confidence:", float(pred[0][0]))