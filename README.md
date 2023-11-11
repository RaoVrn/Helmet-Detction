# Helmet Detection and Person on Motorbike Detection

This project utilizes YOLOv3 models to detect objects related to road safety. The two main components of the project are:

1. **Person on Motorbike Detection**: This part of the project is designed to detect individuals on motorbikes. It can be crucial for ensuring road safety and various applications such as traffic monitoring.

2. **Helmet Detection**: The second component is focused on helmet detection. It is important for enforcing helmet usage, especially for motorbike riders, and improving road safety.

## Datasets

The following datasets were used for training the models:

- [Helmet Detection YOLOv3 Dataset](https://www.kaggle.com/datasets/savanagrawal/helmet-detection-yolov3): This dataset is used for helmet detection training.

- [Person on Motorbike Using YOLOvs3 Dataset](https://www.kaggle.com/datasets/sureshbose2000/detect-person-on-motorbike-using-yolov3): This dataset is utilized for person on motorbike detection.

- [Sample Images](https://www.kaggle.com/datasets/sureshbose2000/pic-detection): A collection of sample images used for demonstration and testing purposes.

## Usage

The project includes a Flask-based API that integrates the trained YOLOv3 models. Users can upload an image through a web interface, and the system will provide object detection results, including helmet detection and person on motorbike detection.

## Requirements

- Python
- OpenCV
- Matplotlib
- Flask (for the web interface)


