# ffnn-binary-classification-keras-
# FFNN Binary Classification with Keras

A binary classification project that uses a Feed Forward Neural Network
(FFNN) built with TensorFlow and Keras.

The project demonstrates how to prepare a labelled dataset, divide it into
training and testing sets, train a neural network, visualise its learning
progress, and evaluate its performance using standard classification
metrics.

## Project Overview

The dataset contains:

- 4,500 labelled examples
- 512 input features per example
- Two output classes: positive and negative

The dataset is divided into:

- 85% training data
- 15% testing data

The split uses stratification to preserve the class distribution between
the training and testing sets.

## Objectives

This project demonstrates how to:

1. Load and prepare a dataset using Pandas.
2. Split data into training and testing sets.
3. Build a Feed Forward Neural Network using Keras.
4. Train the model for binary classification.
5. Visualise training accuracy and loss.
6. Test the model on unseen data.
7. Calculate Precision, Recall, F1-Score, and Accuracy.

## Technologies Used

- Python
- Jupyter Notebook
- Google Colab
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Architecture

The model is a sequential Feed Forward Neural Network containing:

- An input layer accepting 512 features
- One or more fully connected hidden layers
- ReLU activation functions in the hidden layers
- One output neuron with a sigmoid activation function

The sigmoid output represents the predicted probability of the positive
class.

## Training

The model is trained using:

- Epochs: 200
- Batch size: 64
- Loss function: Binary cross-entropy
- Optimiser: [replace with your optimiser, such as Adam]
- Evaluation metric: Accuracy

## Model Evaluation

The trained model is evaluated on the unseen test set using:

| Metric | Result |
|---|---:|
| Precision | Replace with your result |
| Recall | Replace with your result |
| F1-Score | Replace with your result |
| Accuracy | Replace with your result |

### Metric Descriptions

- **Precision:** The proportion of positive predictions that were correct.
- **Recall:** The proportion of actual positive examples correctly identified.
- **F1-Score:** The harmonic mean of precision and recall.
- **Accuracy:** The proportion of all test examples classified correctly.

## Running the Project

### Google Colab

1. Open Google Colab.
2. Upload `FFNN_Classification.ipynb`.
3. Upload `data.csv` to the Colab session.
4. Select **Runtime → Run all**.
5. Review the training graphs and evaluation results.

### Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ffnn-binary-classification-keras.git
cd ffnn-binary-classification-keras
