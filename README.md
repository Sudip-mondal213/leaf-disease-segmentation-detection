# An Innovative Deep Learning Approach for Leaf Image Segmentation and Disease Detection using Mask R-CNN and Small Inception Model

## About The Project

This project was developed as part of my final year work. The main objective is to improve leaf disease detection by first isolating the leaf from the background and then performing classification on the segmented leaf.

The system combines Mask R-CNN for leaf segmentation and a custom Small Inception Model for disease detection.

Instead of directly classifying the full image, the model first removes unnecessary background information and focuses only on the leaf region.

---

## Workflow

```text
Input Leaf Image
        ↓
Mask R-CNN Segmentation
        ↓
Background Removal
        ↓
Segmented Leaf
        ↓
Small Inception Model
        ↓
Healthy / Diseased Prediction
```

---

## Technologies Used

* Python
* PyTorch
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Mask R-CNN

---

## Project Structure

```text
leaf-disease-segmentation-detection/

├── final_project.py
├── small_inception_model.keras
├── README.md

├── sample_images/
│   ├── img1.JPG
│   ├── img2.JPG
│   └── ...

├── notebooks/
│   ├── maskrcnn_training.ipynb
│   └── inception2.ipynb

├── test_maskrcnn.py
└── test_model.py
```

---

## Running The Project

Run the following command:

```bash
python final_project.py image.jpg
```

Example:

```bash
python final_project.py sample_images/img1.JPG
```

The program will:

1. Load the image
2. Segment the leaf
3. Remove the background
4. Classify the leaf
5. Display the final prediction

---

## Sample Images

Several sample leaf images have been included in this repository for testing purposes.

You can use any image from the `sample_images` directory to test the pipeline.

---

## Pretrained Models

The Small Inception classification model is included in this repository.

The Mask R-CNN model file is not included because its size exceeds GitHub's upload limit.
"https://drive.google.com/file/d/1ICCCW52t7g30tET3sHgo3iH3gfORf134/view?usp=drive_link"

To run the complete project, place the pretrained file below in the project root directory:

```text
leaf_maskrcnn.pth
```

---

## Current Limitations

The project is still under development and some issues remain.

### 1. Citrus Greening Detection

Some images belonging to:

```text
Orange___Haunglongbing_(Citrus_greening)
```

are occasionally classified as Healthy.

### 2. False Disease Detection

Young tomato leaves with epidermal hairs are sometimes classified as Diseased even though they are healthy.

### 3. Segmentation Accuracy

Mask R-CNN occasionally fails to segment leaves correctly when:

* leaf edges are heavily damaged
* leaves are folded
* shadows are present
* backgrounds are complex

### 4. Binary Classification

The current version only predicts:

* Healthy
* Diseased

Disease-specific prediction is not yet supported.

---

## Future Improvements

* Improve segmentation accuracy
* Increase dataset size
* Reduce false positive predictions
* Improve Citrus Greening detection
* Add disease-specific classification
* Develop a web-based interface
* Support real-time prediction

---

## Project Status

Work in Progress

Current focus areas:

* Better leaf segmentation
* Better disease classification
* More robust performance on real-world leaf images

---

## Author

Sudip Mondal

Project is lowkey fire
