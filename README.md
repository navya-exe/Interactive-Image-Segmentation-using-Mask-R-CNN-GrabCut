# Interactive Image Segmentation using Mask R-CNN and GrabCut

## Overview

This project implements an interactive image segmentation system using **Mask R-CNN** for instance segmentation and **GrabCut** for mask refinement. The system detects objects in an input image, generates segmentation masks, refines the detected masks using GrabCut, and visualizes the results with bounding boxes, class labels, and confidence scores.

The project demonstrates the application of deep learning and classical computer vision techniques for accurate object segmentation.

---

## Features

- Object Detection using Mask R-CNN
- Instance Segmentation
- Mask Refinement using GrabCut
- Bounding Box Visualization
- Class Label Prediction
- Confidence Score Display
- Segmented Object Extraction
- Overlay Visualization
- Automatic Output Image Generation

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Mask R-CNN (Pre-trained COCO Model)

---

## Project Structure

```
Image_segmntation/
│
├── images/
│   ├── input/
│   │   └── test.jpg
│   │
│   └── output/
│       ├── result.jpg
│       ├── overlay.jpg
│       ├── boxed.jpg
│       └── refined.jpg
│
├── src/
│   ├── main.py
│   ├── segmentation.py
│   └── utils.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Workflow

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
Mask R-CNN
      │
      ▼
Object Detection
      │
      ▼
Instance Segmentation
      │
      ▼
GrabCut Refinement
      │
      ▼
Bounding Boxes & Labels
      │
      ▼
Output Images
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/interactive-image-segmentation.git

cd interactive-image-segmentation
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Place an input image inside:

```
images/input/
```

Run the project:

```bash
python src/main.py
```

---

## Output

The following output images are generated automatically:

| Output | Description |
|---------|-------------|
| result.jpg | Segmented object |
| overlay.jpg | Segmentation overlay on original image |
| boxed.jpg | Bounding box with class label and confidence |
| refined.jpg | GrabCut refined segmentation |

---

## Sample Results

### Original Image

(Add screenshot here)

### Segmented Image

(Add screenshot here)

### Overlay

(Add screenshot here)

### Bounding Boxes

(Add screenshot here)

### GrabCut Refined Result

(Add screenshot here)

---

## Applications

- Autonomous Vehicles
- Medical Image Analysis
- Robotics
- Video Surveillance
- Object Recognition
- Image Editing
- Augmented Reality

---

## Future Scope

- Support segmentation of multiple objects simultaneously.
- Real-time webcam segmentation.
- Custom dataset training.
- Web application deployment using Flask or Streamlit.
- Integration with Segment Anything Model (SAM).

---

## Author

**Ch. Navya Naidu**

B.Tech Computer Science and Engineering (AI & ML)

---

## License

This project is licensed under the MIT License.
