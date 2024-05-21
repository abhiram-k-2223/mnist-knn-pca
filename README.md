# Handwritten Digit Recognition

This project aims to recognize handwritten digits using machine learning models. The dataset used is a handwritten digits dataset, and the target variable is 'label', which indicates the digit. The workflow includes data loading, visualization, preprocessing, and training a K-Nearest Neighbors (KNN) classifier with and without Principal Component Analysis (PCA).

## Project Description

### Data Loading

The dataset is loaded using pandas, and an initial exploration is done using the `head()` method to understand the structure and content of the data.

### Data Visualization

A sample digit is visualized using Matplotlib to gain insight into the dataset's appearance. The digit is reshaped into a 28x28 pixel grid and displayed as an image.

### Feature and Target Separation

1. **Features (X)**: All columns except 'label' are used as features representing the pixel values of the digit images.
2. **Target (y)**: The 'label' column is used as the target variable, representing the digit.

### Data Splitting

The dataset is split into training and testing sets using an 80-20 split with a fixed random state of 42 to ensure reproducibility.

### Model Training and Evaluation

#### K-Nearest Neighbors (KNN) Classifier

1. **Initial Training**: A KNN classifier is trained on the raw pixel values of the training data.
2. **Prediction and Timing**: Predictions are made on the test data, and the time taken for prediction is measured.
3. **Evaluation**: The model's performance is evaluated using the accuracy score, indicating the proportion of correct predictions.

#### Preprocessing with StandardScaler and PCA

1. **Scaling**: The pixel values are scaled using StandardScaler to have zero mean and unit variance, which improves the performance of the PCA and KNN classifier.
2. **Principal Component Analysis (PCA)**: The dimensionality of the data is reduced using PCA, retaining 100 principal components. This step helps in reducing computational complexity and improving model performance.

### Model Training and Evaluation with PCA

1. **Training**: A KNN classifier is trained on the PCA-transformed training data.
2. **Prediction**: Predictions are made on the PCA-transformed test data.
3. **Evaluation**: The model's performance is evaluated using the accuracy score, and the new score is compared to the initial score without PCA.

### Comparison of Models

The accuracy scores of the KNN classifier with and without PCA are compared to determine the effectiveness of dimensionality reduction in improving model performance.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/abhiram-k-2223/mnist-knn-pca.git
    cd mnist-knn-pca
    ```

2. **Install dependencies**:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

3. **Run the Jupyter notebook or script**:
    - If using a Jupyter notebook, open `mnist_dataset.ipynb` and run the cells sequentially.

4. **Output**:
    - Accuracy scores of the KNN classifier with and without PCA, indicating the performance on the test set.
    - Time taken for prediction with the KNN classifier on raw and PCA-transformed data.

This project demonstrates how to preprocess data, visualize samples, and train a KNN classifier with and without PCA for the task of handwritten digit recognition.
