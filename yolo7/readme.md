# YOLOv7 Segmentation Project

This repository provides a step-by-step guide to setting up and using YOLOv7 for segmentation tasks, specifically focused on crack detection.

## Prerequisites

Before you start, ensure you have the following installed:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Python 3.10
- Git

## Setup Instructions

Follow these steps to set up the environment and prepare the YOLOv7 segmentation model:

### 1. Clone YOLOv7 Repository

1. Clone the YOLOv7 repository:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7
   cd yolov7
   ```

2. Checkout the specific commit to ensure compatibility:
   ```bash
   git checkout 44f30af0daccb1a3baecc5d80eae22948516c579
   ```

### 2. Install Dependencies

1. Navigate to the segmentation directory:
   ```bash
   cd seg
   ```

2. Update `pip` and install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 3. Download Pre-trained Weights

Download the YOLOv7 segmentation model's pre-trained weights:
```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt
```

### 4. Set Up the Environment

1. Navigate to your project directory:
   ```bash
   cd Dev/crack_seg_yolov7/yolov7/seg/
   ```

2. Activate the Conda environment:
   ```bash
   conda activate crack_seg_py310
   ```

3. Compile or configure any additional dependencies (if applicable):
   ```bash
   cc
   ```

### 5. Training on Google Colab

You can train the model using Google Colab by following this [notebook](https://colab.research.google.com/drive/14ItyKLiLMmzMHOZL7Orh6E61NJSNQ2W6#scrollTo=U2Eaudcz4L5a). After training:

1. Download the `data.yaml` and `best.pt` files.
2. Place them in the appropriate directories within your project.

### 6. Download and Prepare the Dataset

Run the provided script to download and prepare your dataset:
```bash
python download_dataset.py
```

### 7. Run the Segmentation Prediction

After setting everything up, run the segmentation prediction using the following command:
```bash
python segment/predict.py --weights /home/rafael2ms/Dev/crack_seg_YoloV7/yolov7/pretrained/best.pt --conf 0.25 --source /home/rafael2ms/Dev/crack_seg_YoloV7/yolov7/seg/RIGID-PAVEMENT-2/test/images --name custom
```

## Additional Resources

For further learning and resources on YOLO object detection, visit [this blog post](https://www.v7labs.com/blog/yolo-object-detection).
