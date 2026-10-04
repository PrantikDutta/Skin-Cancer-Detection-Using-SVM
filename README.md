# Skin Cancer Detection Using SVM

A machine learning project for classifying skin lesions using the HAM10000 dermatoscopic image dataset and a Support Vector Machine (SVM) classifier.

The main goal of this project is to explore how traditional image-processing and machine-learning techniques can be used for skin lesion classification. Instead of using a deep learning model, this project uses Histogram of Oriented Gradients (HOG) features combined with an RBF-kernel SVM.

## Dataset

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset. It contains **10,015 dermatoscopic images** covering seven different categories of skin lesions.

The dataset can be accessed through Kaggle:

**Kaggle:**  
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset is not included in this repository because of its size. Download the dataset from Kaggle and place the required files in the project directory before running the notebook.

The seven diagnostic categories are:

- `akiec` - Actinic keratoses
- `bcc` - Basal cell carcinoma
- `bkl` - Benign keratosis-like lesions
- `df` - Dermatofibroma
- `mel` - Melanoma
- `nv` - Melanocytic nevi
- `vasc` - Vascular lesions

## Approach

The project follows these main steps:

1. Load the HAM10000 metadata and images.
2. Resize images to `128 × 128`.
3. Normalize pixel values.
4. Split the original images into training and testing sets using a stratified split.
5. Apply data augmentation only to the training images.
6. Extract HOG features from the images.
7. Train an RBF-kernel SVM classifier.
8. Evaluate the model using accuracy, ROC-AUC, and other classification metrics.

## Data Augmentation

Data augmentation is applied only after the train-test split. This prevents augmented versions of test images from being included in the training data.

The following transformations are used:

- Rotation
- Width shifting
- Height shifting
- Horizontal flipping

Three augmented images are generated for each original training image.

The resulting dataset contains approximately:

- **Training images:** 32,048
- **Testing images:** 2,003

The test set is kept completely separate and is not augmented.

## Feature Extraction

The project uses **Histogram of Oriented Gradients (HOG)** to extract features from the dermatoscopic images.

The HOG configuration is:

```text
Image size:          128 × 128
Orientations:        9
Pixels per cell:     16 × 16
Cells per block:     2 × 2
