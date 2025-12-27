# Amazon Fake Review Detection (DTS402)

This project implements a binary classification system for detecting fake Amazon reviews using a custom-built Multi-Layer Perceptron (MLP). The project demonstrates a complete machine learning pipeline, covering data preprocessing, feature engineering (TF-IDF + SVD), strict data leakage prevention (splitting before embedding), model construction, and final evaluation.

## Project Overview

The objective is to distinguish between **Original/Real Reviews (OR)** and **Computer Generated/Fake Reviews (CG)**.
By combining an open-source dataset with a self-collected dataset, we build a robust detection model. Key highlights include a preprocessing workflow strictly designed to prevent data leakage and a hand-coded neural network implementation featuring Batch Normalization and Dropout.

## Directory Structure

```
DTS402/
├── dataset/                  # Stores raw and processed CSV data files
├── output/                   # Stores generated charts and model files
│   └── model/                # Saved .pkl model weights
├── Sample_and_Shuffle.py     # 1. Sampling from the open-source dataset
├── Dataset_compare.py        # 2. Dataset comparison and EDA
├── Data_Cleaning.py          # 3. Cleaning the self-collected dataset
├── Merge.py                  # 4. Merging datasets
├── Featurization_save.py     # 5. Basic feature extraction
├── Split.py                  # 6. Train/Test split (Crucial for leakage prevention)
├── Data_Embedding.py         # 7. Text Vectorization (TF-IDF + SVD)
├── Model.py                  # 8. MLP Model class definition
├── TrainingData_Prepare.py   # 9. Training data loading and standardization
├── Train_and_Evaluation.py   # 10. Model training and evaluation
└── Compare.py                # 11. Final comparison across datasets
```

## Requirements

Ensure the following Python libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn
```

## Execution Pipeline

Please run the Python scripts in the exact order below to ensure the correct data flow:

### Phase 1: Data Preparation & Preprocessing

1.  **`Sample_and_Shuffle.py`**
    *   **Function**: Randomly samples 3,000 items from the original large-scale dataset (`fake reviews dataset.csv`).
    *   **Output**: `dataset/open_3k.csv`

2.  **`Dataset_compare.py`**
    *   **Function**: Compares distributions (e.g., ratings, text length) between the open-source dataset and the self-collected dataset (`self_dataset.csv`) using EDA techniques.
    *   **Output**: Visualization charts for comparison.

3.  **`Data_Cleaning.py`**
    *   **Function**: Cleans the self-collected dataset by unifying column names and fixing encoding issues (e.g., replacing curly quotes) to match the open-source format.
    *   **Output**: `dataset/self_dataset_cleaned.csv`

4.  **`Merge.py`**
    *   **Function**: Merges the processed open-source data and self-collected data into a single dataset.
    *   **Output**: `dataset/merged_dataset.csv`

### Phase 2: Feature Engineering

5.  **`Featurization_save.py`**
    *   **Function**: Extracts basic text features (e.g., `word_count`, `excl_count` for exclamation marks).
    *   **Output**: `dataset/merged_dataset_with_features.csv`

6.  **`Split.py`** (Critical Step)
    *   **Function**: Splits the dataset into Training (80%) and Test (20%) sets *before* vectorization.
    *   **Purpose**: **Prevents Data Leakage**, ensuring the test set remains completely unseen during feature construction.
    *   **Output**: `dataset/train_raw.csv`, `dataset/test_raw.csv`

7.  **`Data_Embedding.py`**
    *   **Function**:
        *   Builds the vocabulary and computes the TF-IDF matrix using **only the Training Set**.
        *   Applies SVD (Singular Value Decomposition) to reduce text features to 24 dimensions.
        *   Transforms the **Test Set** using the parameters (Vocabulary, IDF, SVD projection) learned from the training set.
    *   **Output**: `dataset/train_processed_final.csv`, `dataset/test_processed_final.csv`

### Phase 3: Model Training & Evaluation

8.  **`Model.py`**
    *   **Function**: Defines the `SimpleMLP` class.
    *   **Architecture**: Input -> Linear(64) -> BN -> ReLU -> Dropout -> Linear(32) -> BN -> ReLU -> Dropout -> Linear(1) -> Sigmoid.
    *   **Optimizer**: SGD with Momentum.
    *   **Note**: This is a module file imported by subsequent scripts; do not run it directly.

9.  **`TrainingData_Prepare.py`**
    *   **Function**: Loads the processed train/test files and performs final feature Normalization.
    *   **Note**: This is a data loading module imported by `Train_and_Evaluation.py`.

10. **`Train_and_Evaluation.py`**
    *   **Function**:
        *   Instantiates and trains the MLP model.
        *   Monitors Loss and Precision in real-time.
        *   Saves the best model weights (`model_best.pkl`).
        *   Generates evaluation charts: SVD visualization, Loss curves, Confusion Matrix, etc.
    *   **Output**: Model files and charts in the `output/` directory.

11. **`Compare.py`**
    *   **Function**: Loads the best trained model and evaluates its performance separately on the "Open Dataset," "Self-collected Dataset," and the "Mixed Dataset."
    *   **Output**: Performance comparison bar charts and detailed metric tables.

## Model Architecture Details

This project implements a neural network from scratch using `numpy` (without PyTorch/TensorFlow), incorporating modern deep learning techniques:

*   **Input Layer**: 27 Dimensions (1 Rating + 2 Basic Text Features + 24 SVD Features).
*   **Hidden Layer 1**: 64 Neurons (He Initialization) + Batch Normalization + ReLU + Inverted Dropout.
*   **Hidden Layer 2**: 32 Neurons + Batch Normalization + ReLU + Inverted Dropout.
*   **Output Layer**: 1 Neuron (Sigmoid Activation).
*   **Loss Function**: Binary Cross-Entropy.
*   **Regularization**: L2 Regularization (Weight Decay).

## Notes

*   **File Paths**: The scripts use hardcoded absolute paths (e.g., `D:\Projects\DTS402\...`). Please update these paths to match your local environment before running the code.
*   **Fonts**: The plotting code uses `SimHei` to support Chinese characters. If you are not on Windows or lack this font, you may need to adjust the font settings in the visualization scripts.

## Author
Zijie Xue 2575576
```