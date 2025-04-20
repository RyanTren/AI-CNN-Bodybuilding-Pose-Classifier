# Bodybuilding Pose Classification Using Transfer Learning

## 1. Introduction

This project focuses on developing an AI-based image processing model to classify bodybuilding poses. Utilizing transfer learning with MobileNetV2, the model aims to accurately identify five fundamental bodybuilding poses, enhancing applications in competition judging, athlete training, and pose analysis. The primary objective is to create a reliable and efficient model capable of classifying bodybuilding poses with high accuracy, even with a limited dataset. By employing transfer learning, the project seeks to reduce training time and computational resources while maintaining performance.


## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Overview](#2-project-overview)
3. [Implementation Details](#4-implementation-details)
4. [Data Collection](#5-data-collection)
5. [Model Training](#6-model-training)
6. [Known Limitations](#7-known-limitations)
7. [Comparison with Alternative Model](#8-comparison-with-alternative-model)
8. [References](#9-references)

---

## 2. Project Overview

The model is designed to classify the following bodybuilding poses:

- Front Double Bicep
- Back Double Bicep
- Side Chest
- Abs and Thighs
- Negative Samples (e.g., non-poses)

Key features include:

- **Transfer Learning**: Leveraging MobileNetV2 pre-trained on ImageNet for efficient feature extraction.
- **Data Augmentation**: Applying techniques like rotation, zoom, and flipping to enhance model robustness.
- **Confidence Scoring**: Implementing thresholds to evaluate pose quality and model certainty.

---

## 3. Implementation Details

- **Framework**: TensorFlow with Keras API.
- **Base Model**: MobileNetV2 (pre-trained on ImageNet).
- **Custom Layers**:
  - Global Average Pooling
  - Dense Layer with ReLU activation
  - Dropout for regularization
  - Output Dense Layer with Softmax activation
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Accuracy.

---

## 4. Data Collection

A custom data collection script was developed to automate the process:

- **Webcam Integration**: Captures images directly from the webcam.
- **User Interaction**:
  - Menu displayed before each capture session.
  - High-pitched beep signals the start and end of image capture.
  - Medium-pitched beeps for countdown.
- **Randomized Filenames**: Appends a unique 10-character alphanumeric string to each image filename to prevent overwriting.
- **Data Storage**: Images are saved in class-specific directories within the `bb_training_data` folder.

---

## 5. Model Training

- **Image Preprocessing**:
  - Resizing to 224x224 pixels.
  - Normalization to scale pixel values between 0 and 1.
- **Data Generators**:
  - Utilized `ImageDataGenerator` for real-time data augmentation.
  - Applied a validation split of 20%.
- **Training Configuration**:
  - Batch Size: 32
  - Epochs: 10
  - Early stopping and model checkpointing implemented to prevent overfitting.
- **Class Weights**:
  - Adjusted to address class imbalance, particularly reducing the weight of the 'not a pose' class to mitigate overprediction.

---

## 6. Known Limitations

- **Class Imbalance**: Despite class weighting, the model may still favor the 'not a pose' class if not adequately balanced.
- **Limited Dataset**: A relatively small number of images per class may affect generalization.
- **Lighting and Background Variations**: Model performance can degrade with significant variations in lighting and backgrounds not represented in the training data.
- **Practical Limitations**: Most of the data was trained using a the same solo male posing in indoor lighting. The model may perform unexpectedly in different lighting conditions or with female competitors.

---

## 7. Comparison with Alternative Model

| Feature                     | Transfer Learning Model  | Custom CNN Model  |
|-----------------------------|----------------------------------------|--------------------------------|
| **Base Architecture**       | MobileNetV2 (Pre-trained)              | Custom-built CNN               |
| **Training Time**           | Shorter due to pre-trained layers      | Longer, trained from scratch   |
| **Data Requirement**        | Less data needed                      | Requires more data             |
| **Performance**             | Higher initial accuracy                | Potential for higher accuracy with sufficient data |
| **Flexibility**             | Limited to MobileNetV2 architecture    | Fully customizable             |
| **Complexity**              | Easier to implement                    | More complex implementation    |

---

## 8. References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications - MobileNetV2](https://keras.io/api/applications/mobilenet/)
- [ImageDataGenerator Documentation](https://keras.io/api/preprocessing/image/)
