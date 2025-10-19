# SCT_ML_4
# Hand Gesture Recognition using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to accurately identify and classify **hand gestures** from images, enabling intuitive human-computer interaction and gesture-based control systems. It is part of **Task 4** of the **SkillCraft Technology Internship**.

The project demonstrates a complete **deep learning pipeline**, from data preprocessing to model training, evaluation, and real-time prediction.

## Features
- **Image loading and preprocessing**:
  - Grayscale conversion
  - Resizing to 64x64
  - Normalization
  - One-hot encoding of gesture labels
- **Dataset splitting** into training and test sets
- **CNN architecture**:
  - Multiple convolutional and pooling layers
  - Fully connected layers with dropout
  - Softmax output for multi-class classification
- **Model training** with accuracy and loss monitoring
- **Evaluation**:
  - Test accuracy
  - Confusion matrix visualization
  - Sample predictions with color-coded correctness
- **Model persistence**:
  - Save trained CNN model
  - Save label encoder
  - Load and use model for inference
- **Real-time gesture prediction** using image uploads

## Technologies Used
- Python  
- OpenCV, NumPy, Matplotlib, Seaborn  
- TensorFlow, Keras  
- scikit-learn (LabelBinarizer, train_test_split)

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
## Dataset

Download the **Sign Language Digits** dataset from GitHub: [https://github.com/ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)  

After downloading, extract the `Dataset/` folder and place it in the project root so that the code can access the images correctly.
