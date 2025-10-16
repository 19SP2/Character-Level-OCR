# Character-Level-OCR

A deep learning project for Optical Character Recognition (OCR) that recognizes individual characters and short text sequences from synthetic and handwritten-style images using a Convolutional Recurrent Neural Network (CRNN) architecture trained with CTC loss. This project implements a Character-Level OCR system trained on synthetic datasets generated using the TextRecognitionDataGenerator (TRDG) and EMNIST datasets. It’s designed to recognize small sequences of alphanumeric characters and punctuation, similar to text seen in handwritten notes or scanned forms.

## Model Architecture

The OCR model is based on the CRNN (Convolutional Recurrent Neural Network):
- CNN layers extract spatial features from the image.
- Bi-directional LSTM layers learn the sequence dependencies.
- CTC loss is used to align variable-length text outputs with predictions.

## Dataset Preparation

Run this to generate synthetic OCR training data and prepare it for training: python prepare_ocr_dataset.py

This script:
- Creates datasets using TRDG (TextRecognitionDataGenerator).
- Combines it with EMNIST or other sources.
- Saves image paths and labels to a CSV file: datasets/data.csv.

## Training

Train the model using your custom dataset: python train_crnn.py
After training, the model and character mapping are saved in: ocr_crnn_model/crnn.pth

## Evaluation

During or after training, you’ll see logs like:
Epoch 200/200 — Loss: 0.1245 — Accuracy: 95.38%
✅ Model saved to ocr_crnn_model/crnn.pth

## Inference / Testing

To predict text from new images: python inference_crnn.py

Example output:
Image: datasets/emnist_images/00000.png
Ground Truth: Z
Predicted   : 2

Image: datasets/emnist_images/00001.png
Ground Truth: a
Predicted   : a

## Folder Structure

OCR Project/ 
├── myFonts/ 
│ └── DejaVuSans.ttf 
├── datasets/ 
│ ├── emnist/ 
│ ├── trdg_custom/ 
│ ├── synthetic_math/ 
│ └── data.csv 
├── prepare_ocr_dataset.py 
├── train_model.py 
├── inference_crnn.py 
├── ocr_crnn_model/ 
│ └── crnn.pth 
└── venv/

## Acknowledgments

- TextRecognitionDataGenerator (TRDG)
- EMNIST Dataset
- Synthetic Math Symbol Dataset

🧩 Author: Sharvari Sunil Pradhan
📘 Project: Character-Level OCR (CRNN)
📅 Year: 2025
✨ Open for collaboration and improvements!
