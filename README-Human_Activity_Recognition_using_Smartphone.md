# Human Activity Recognition Using Smartphones

Classifying human physical activity (walking, standing, sitting, and more) from smartphone accelerometer and gyroscope readings, comparing a from-scratch statistical classifier against a neural network.

![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20TensorFlow%2FKeras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-blue)

## Overview

The goal of this project is to build an accurate model that classifies a person's activity from raw smartphone sensor data — specifically, distinguishing between six activities: **WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING,** and **LAYING**.

It uses the [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) from the UCI Machine Learning Repository — ~10,299 instances of 3-axial linear acceleration and 3-axial angular velocity readings, pre-split into training and test sets, with 561 engineered time- and frequency-domain features per sample.

This was a group project (Group 14) for a statistical learning course at Northeastern University.

## Approach

1. **Data validation** — confirmed the training and test sets have zero missing values.
2. **Exploratory data analysis** — visualized the class balance of activities in both the training and test splits.
3. **Dimensionality reduction** — applied PCA (retaining 95% of variance) to the standardized 561-feature space before modeling.
4. **Modeling**, with three classifiers built and compared:
   - **Gaussian Naive Bayes** — implemented from scratch (custom class, not `sklearn`).
   - **Softmax (multinomial logistic) Regression** — also implemented from scratch, with gradient-descent training.
   - **Neural Network** — a Keras/TensorFlow `Sequential` model (dense layers with dropout), trained for 10 epochs.
5. **Evaluation** — accuracy, macro precision/recall/F1, and a full classification report + confusion matrix for the neural network.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Gaussian Naive Bayes (from scratch) | 80.2% | 0.797 |
| Softmax Regression (from scratch) | 94.3% | 0.942 |
| Neural Network (Keras) | **95.0%** | 0.95 |

The neural network was the strongest performer, with especially clean separation on `LAYING` (100% precision/recall) and the most confusion between the dynamic `SITTING`/`STANDING` pair — a known hard case for this dataset since those two postures produce very similar static sensor signatures.

## Repository Contents

| File | Description |
|---|---|
| `StatisticalLearningProject_Final.ipynb` | Main analysis notebook — data loading, EDA, PCA, and all three models. |
| `Project Group 14 - Human Activity Recognition Using Smartphones.pdf` | Initial project plan: goal, dataset description, and methodology. |
| `Group 14 project report_final.pdf` | Final written project report. |
| `Final_PPT.pptx` | Final presentation deck. |

## Tech Stack

- Python — `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` — preprocessing, `train_test_split`, `StandardScaler`, `PCA`
- `TensorFlow` / `Keras` — neural network model
- Originally developed in Google Colab

## Getting Started

The notebook was built for Google Colab and expects the UCI HAR dataset files (`X_train.txt`, `y_train.txt`, `X_test.txt`, `y_test.txt`) to be uploaded at runtime. To run it locally instead:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tqdm
```

1. Download the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and place `X_train.txt`, `y_train.txt`, `X_test.txt`, and `y_test.txt` in your working directory.
2. Replace the Colab `files.upload()` cell with local file paths.
3. Run `StatisticalLearningProject_Final.ipynb` top to bottom.

## Team

Group 14: Raaga Sindhu Mangalagiri, Shriram Vijaykumar, Prajwal Srinivas, Siva Vasanta Harika Mangu, and Varun Kumar Kumaravel.

## License

No license is currently specified for this repository.
