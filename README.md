
# Computer Vision: Image Classification & Object Detection

This repository hosts a collection of deep learning projects focusing on Image Classification and Object Detection. It leverages state-of-the-art frameworks like TensorFlow/Keras and PyTorch (including YOLOv5 and Faster R-CNN) to solve computer vision tasks on custom datasets.

## 📂 Repository Structure

The project is organized into distinct notebooks, each tackling a specific problem or architecture:

| Notebook Name | Task | Framework | Description |
|---------------|------|-----------|-------------|
| `CustomResNet.ipynb` | Classification | TensorFlow/Keras | A custom Residual Network (ResNet) implementation built from scratch to classify images. |
| `ResNet50.ipynb` | Classification | TensorFlow/Keras | Transfer learning using the pre-trained ResNet50 architecture for robust image classification. |
| `FRCNN.ipynb` | Object Detection | PyTorch | Implementation of Faster R-CNN with a MobileNetV3-Large FPN backbone for detecting object classes. |
| `YOLO.ipynb` | Object Detection | PyTorch (Ultralytics) | Application of the YOLOv5 (You Only Look Once) model for real-time object detection. |

## Features

- **Image Classification**: Custom and pre-trained ResNet architectures
- **Object Detection**: Faster R-CNN and YOLOv5 implementations
- **Transfer Learning**: Leveraging pre-trained models for improved performance
- **Custom Datasets**: Trained on domain-specific image data

## Technologies Used

- **Frameworks**: TensorFlow/Keras, PyTorch
- **Models**: ResNet50, Faster R-CNN, YOLOv5
- **Tools**: Jupyter Notebooks, Python

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x
- Ultralytics YOLOv5
- OpenCV
- NumPy, Matplotlib

## Usage

Each notebook is self-contained and can be run independently:
```bash
jupyter notebook CustomResNet.ipynb
```

Follow the instructions within each notebook to train models or run inference.

## Project Highlights

- Custom ResNet implementation demonstrating deep understanding of CNN architectures
- Transfer learning techniques for efficient model training
- Comparative analysis of detection frameworks (Faster R-CNN vs. YOLO)
- Real-time object detection capabilities with YOLOv5
