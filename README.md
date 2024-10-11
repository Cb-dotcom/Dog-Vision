# 🐶 End-to-End Multi-Class Dog Breed Classification

This repository contains an end-to-end implementation for multi-class dog breed classification using TensorFlow 2.x and TensorFlow Hub. The goal is to accurately identify the breed of a dog given an image. The project is structured into multiple stages, starting with model creation and initial training on Google Colab. Future stages will include deployment, containerization, and cloud integration.

## Table of Contents
- [Project Overview](#project-overview)
- [Step 1: Model Training on Colab](#step-1-model-training-on-colab)
  - [Problem Statement](#problem-statement)
  - [Dataset](#dataset)
  - [Evaluation](#evaluation)
  - [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
- [Making Predictions on Custom Images](#making-predictions-on-custom-images)
- [Future Work](#future-work)
- [Using Git Branches](#using-git-branches)
- [Key Learnings](#key-learnings)

---

## Project Overview
This project aims to create a deep learning-based model capable of classifying dog breeds from an image. Leveraging transfer learning with TensorFlow Hub's MobileNetV2, a pre-trained model, the training process is accelerated and accuracy is enhanced. The project is divided into various steps for better tracking and deployment.

## Step 1: Model Training on Colab
### Problem Statement
The objective is to identify the breed of a dog based on a given image. For example, if a picture of a dog is taken at a cafe, the model should predict the dog's breed with high confidence.

### Dataset
- **Source**: [Kaggle's Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification/data)
- **Classes**: 120 different dog breeds
- **Training Data**: 10,000+ labeled images
- **Test Data**: 10,000+ unlabeled images

### Evaluation
The model is evaluated based on breed prediction probabilities for each test image, submitted as a CSV file for the Kaggle competition. More details can be found [here](https://www.kaggle.com/c/dog-breed-identification/overview/evaluation).

### Features
- **Unstructured Data Handling**: Processes image data using deep learning techniques.
- **Transfer Learning**: Fine-tunes a pre-trained model (MobileNetV2) for this specific classification problem.
- **Batch Processing**: Optimizes memory usage and training speed by splitting data into batches.

## Project Structure
- **Step 1**: Initial model training on Colab
- **Step 2**: Model deployment as an API using FastAPI or Flask
- **Step 3**: Containerization with Docker
- **Step 4**: Deployment on cloud platforms (e.g., AWS, Google Cloud, Heroku)
- **Step 5**: Frontend integration for user interaction

## Setup and Installation
### Google Colab Setup
- The Colab notebook for this project can be found in the `/colab_notebooks` directory.
- **Ensure GPU is enabled** for faster processing.

### Installation
Install the required libraries:
```bash
!pip install tensorflow==2.17.0 tensorflow_hub==0.16.1 matplotlib pandas numpy
