# 🦴 Bone Fracture Classifier using Deep Learning

This project focuses on building a deep learning model to classify bone fractures from X-ray images using a convolutional neural network (CNN). The dataset used contains multi-region X-ray images categorized into `fractured` and `not_fractured`.

## 📂 Dataset

- **Source:** [Fracture Multi-Region X-Ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
- The dataset is structured into three splits:
  ```
  dataset_bone_fracture/
  ├── train/
  │   ├── fractured/
  │   └── not_fractured/
  ├── val/
  │   ├── fractured/
  │   └── not_fractured/
  └── test/
      ├── fractured/
      └── not_fractured/
  ```

**Note:** The dataset is large and is not included in this repository. Please download it from Kaggle and place the unzipped `dataset_bone_fracture` folder in the same directory as the notebook.

## 📊 Exploratory Data Analysis

- Directory structure and file counts explored using a custom `explore_dataset()` function.
- Grid visualization of sample images from each class.
- File integrity checks to remove any corrupted images.
- Distribution plots showing image counts across train, val, and test splits.

## 🧪 Preprocessing and Augmentation

- All images resized to **`224x224`**.
- Image data generators apply random transformations:
  - Rotation, width/height shift, zoom, horizontal/vertical flip.
- Pixel values normalized to the `[0, 1]` range.

## 🧠 Model Architecture

- Sequential CNN with:
  - Convolutional layers + MaxPooling
  - Batch Normalization and Dropout
  - Dense layers with ReLU and softmax activation
- Compiled with:
  - **Loss:** Categorical Crossentropy
  - **Optimizer:** Adam
  - **Metrics:** Accuracy

## 📈 Training and Evaluation

- Trained using:
  ```python
  history = model.fit(
      train_data,
      validation_data=val_data,
      epochs=10,
      steps_per_epoch=len(train_data),
      validation_steps=len(val_data)
  )
  ```
- Visualizations:
  - **Accuracy and Loss vs. Epoch**
  - **Confusion Matrix**
  - **Classification Report**

## ✅ Results

- Evaluation on the test set shows classification performance using precision, recall, and F1-score.
- The confusion matrix helps visualize the model's performance in classifying both `fractured` and `not_fractured` images.

## 🚀 How to Run

1. Download the dataset from Kaggle.
2. Unzip it and place the `dataset_bone_fracture` folder in the same directory as the notebook.
3. Install required packages:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn
   ```
4. Run all cells in the notebook.

## 📌 Acknowledgements

- Dataset provided by [bmadushanirodrigo on Kaggle](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
