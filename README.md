# Skin Cancer Detection Using SVM

This is a machine learning project I built to classify skin lesions from dermatoscopic images. It uses the HAM10000 dataset, HOG features, and an RBF-kernel Support Vector Machine. The goal was to see how far traditional image features and a classic classifier can get on a medical imaging problem, without reaching for deep learning.

## Dataset

The project uses **HAM10000 (Human Against Machine with 10000 training images)**, a collection of 10,015 dermatoscopic images covering seven types of skin lesions.

You can download it from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset isn't included in this repo because of its size, so you'll need to download it yourself before running anything.

The seven lesion categories are:

- `akiec`: Actinic keratoses
- `bcc`: Basal cell carcinoma
- `bkl`: Benign keratosis-like lesions
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic nevi
- `vasc`: Vascular lesions

## How It Works

The pipeline is fairly simple:

1. Load the HAM10000 metadata and images.
2. Resize every image to 128 x 128.
3. Normalize the pixel values.
4. Split the data into train and test sets, keeping class proportions the same (stratified split).
5. Augment the training set only.
6. Extract HOG features from the images.
7. Train an RBF-kernel SVM.
8. Evaluate with accuracy and ROC-AUC.

### Data Augmentation

I applied augmentation only after the train-test split. Doing it before would let slightly altered copies of the same image end up in both sets, which leaks information and inflates the results.

The augmentations are:

- Rotation
- Width shifting
- Height shifting
- Horizontal flipping

Each original training image produces three augmented versions. After augmentation, the dataset looks like this:

- **Training images:** 32,048
- **Test images:** 2,003

The test set is left completely untouched.

## HOG Feature Extraction

Histogram of Oriented Gradients (HOG) turns each image into a numeric feature vector based on edge and gradient patterns. The settings I used:

```text
Image size:       128 x 128
Orientations:     9
Pixels per cell:  16 x 16
Cells per block:  2 x 2
```

This gives **1,764 features per image**, so the final feature matrices are:

```text
Training set: (32048, 1764)
Test set:     (2003, 1764)
```

## Model

The classifier is an RBF-kernel SVM from scikit-learn:

```python
SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)
```

## Results

| Metric         |     Result |
| -------------- | ---------: |
| Test Accuracy  | **66.92%** |
| Binary ROC-AUC |  **0.709** |
| Macro ROC-AUC  |  **0.779** |

These numbers aren't meant to be state of the art. They're a baseline showing what traditional features plus a standard classifier can do on this dataset, and a starting point for improvements.

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- scikit-image
- TensorFlow / Keras
- Pillow
- Matplotlib
- Joblib
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/PrantikDutta/Skin-Cancer-Detection-Using-SVM.git
cd Skin-Cancer-Detection-Using-SVM
```

Install the dependencies:

```bash
pip install numpy pandas scikit-learn scikit-image tensorflow pillow matplotlib joblib
```

Or, if you prefer, use the requirements file:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download HAM10000 from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Make sure you have these files:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

Then update the dataset path in the notebook to point to where you saved them:

```python
BASE_PATH = "/path/to/skin cancer"
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. It covers:

- Loading the dataset
- Preprocessing the images
- Stratified train-test splitting
- Data augmentation
- HOG feature extraction
- SVM training
- Model evaluation
- Saving the trained model

## Project Structure

```text
Skin-Cancer-Detection-Using-SVM/
│
├── notebooks/
│   └── skin_cancer_svm.ipynb
│
├── models/
│   └── skin_cancer_svm_model.pkl
│
├── README.md
└── requirements.txt
```

## Limitations

This project is for learning and research, and it has some clear limits:

- It was trained and tested only on HAM10000.
- HOG features can miss visual details that matter in dermatoscopic images.
- The dataset is heavily imbalanced across classes.
- Results can change depending on preprocessing and how the data is split.
- The model has not been clinically validated.
- It should never be used for medical diagnosis or clinical decisions.

## Future Improvements

Things I'd like to try next:

- More thorough SVM hyperparameter tuning
- PCA to reduce the number of features
- Class weighting or resampling to deal with imbalance
- Comparing HOG features against CNN-based features
- Comparing the SVM against deep learning models
- Testing on independent datasets
- Improving detection of malignant lesions through class balancing and threshold tuning

## Dataset Citation

If you use the HAM10000 dataset, please cite:

> Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device, and it should not be used to diagnose skin cancer or guide any clinical decision. If you're worried about a skin lesion, please see a qualified healthcare professional.

## Author

**Prantik Dutta**

GitHub: [https://github.com/PrantikDutta](https://github.com/PrantikDutta)
