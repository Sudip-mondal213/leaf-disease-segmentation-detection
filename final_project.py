import sys
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from torchvision.models.detection import maskrcnn_resnet50_fpn


# ==========================
# IMAGE PATH FROM CMD
# ==========================

if len(sys.argv) < 2:
    print("Usage: python final.py <image_path>")
    sys.exit()

img_path = sys.argv[1]


# ==========================
# LOAD SEGMENTATION MODEL
# ==========================

model = maskrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features

mask_in_features = (
    model.roi_heads.mask_predictor.conv5_mask.in_channels
)

model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        2
    )
)

model.roi_heads.mask_predictor = (
    torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        mask_in_features,
        256,
        2
    )
)

model.load_state_dict(
    torch.load(
        "leaf_maskrcnn.pth",
        map_location="cpu"
    )
)

model.eval()

print("Segmentation Model Loaded")


# ==========================
# LOAD CLASSIFICATION MODEL
# ==========================

classifier = load_model(
    "small_inception_model.h5"
)

print("Classification Model Loaded")


# ==========================
# LOAD IMAGE
# ==========================

img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    sys.exit()

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

img_tensor = torch.tensor(
    img_rgb / 255.0,
    dtype=torch.float32
).permute(2, 0, 1)


# ==========================
# SEGMENTATION
# ==========================

with torch.no_grad():
    prediction = model([img_tensor])[0]

scores = prediction["scores"].numpy()

best_idx = np.argmax(scores)

mask = prediction["masks"][
    best_idx,
    0
].numpy()

mask = (mask > 0.7).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

segmented = img_rgb.copy()

segmented[mask == 0] = [0, 0, 0]

print("Segmentation Completed")


# ==========================
# CLASSIFICATION
# ==========================

cls_img = cv2.resize(
    segmented,
    (32, 32)
)

cls_img = cls_img / 255.0

cls_img = np.expand_dims(
    cls_img,
    axis=0
)

pred = classifier.predict(
    cls_img,
    verbose=0
)

if pred[0][0] > 0.5:

    result = "Diseased"

else:

    result = "Healthy"

confidence = float(pred[0][0])


# ==========================
# OUTPUT
# ==========================

print("\n====================")
print("Prediction :", result)
print("Confidence :", confidence)
print("====================")


# ==========================
# SHOW IMAGES
# ==========================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title(
    f"{result}\nConfidence={confidence:.4f}"
)
plt.axis("off")

plt.show()