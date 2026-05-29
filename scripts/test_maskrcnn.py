import sys
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn


# ---------- Get Image Path from CMD ----------
if len(sys.argv) < 2:
    print("Usage: python test_maskrcnn.py <image_path>")
    sys.exit()

img_path = sys.argv[1]


# ---------- Load Model ----------
model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, 2
)

model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    mask_in_features, 256, 2
)

model.load_state_dict(torch.load("leaf_maskrcnn.pth"))
model.eval()


# ---------- Load Image ----------
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)


# ---------- Predict ----------
# ---------- Predict ----------
with torch.no_grad():
    prediction = model([img_tensor])[0]

# Get scores
scores = prediction['scores'].numpy()

# Select best detection
best_idx = np.argmax(scores)

# Get best mask
mask = prediction['masks'][best_idx, 0].numpy()

# Threshold
mask = (mask > 0.8).astype(np.uint8)

# Morphological cleanup
kernel = np.ones((7,7), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Largest contour only
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:
    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(mask)

    cv2.drawContours(
        clean_mask,
        [largest],
        -1,
        1,
        thickness=cv2.FILLED
    )

    mask = clean_mask

# ---------- Apply Black Background ----------
segmented = img_rgb.copy()
segmented[mask == 0] = [0, 0, 0]


# ---------- Show ----------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(segmented)
plt.title("Segmented")

plt.show()