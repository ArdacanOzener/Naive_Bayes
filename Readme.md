# Bayesian Classifier

## Overview

The `Bayesian` class implements a Naive Bayes classifier, designed for training and testing on datasets stored in JSON format. It calculates prior probabilities, likelihoods with Laplace smoothing, and evaluates performance using metrics like accuracy, precision, recall, specificity, and F1-score.

---

## Features

1. **Training**: Calculates priors and likelihoods from a training dataset.
2. **Testing**: Predicts class labels for test data and evaluates the model's performance.
3. **Model Persistence**: Save and load model parameters (priors and likelihoods) to/from a JSON file.
4. **Dataset Summary**: Displays detailed summaries of training and testing datasets, including attribute statistics and class distributions.

---

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `json`

Install dependencies using:
```bash
pip install pandas
```

---

## Class Methods

### Constructor: `__init__(train_dataset=None, test_dataset=None)`
Initializes the `Bayesian` class with training and testing datasets.

- **Parameters**:
  - `train_dataset` (str): File path to the training dataset in JSON format.
  - `test_dataset` (str): File path to the testing dataset in JSON format.
  
- **Output**:
  - Displays dataset summaries for training and testing datasets.

---

### `train()`
Trains the Naive Bayes classifier by calculating:
1. **Prior probabilities**: The likelihood of each class in the dataset.
2. **Likelihoods**: Conditional probabilities of feature values given each class, using Laplace smoothing.

---

### `get_dataset(filepath)`
Loads a dataset from a JSON file and returns it as a Pandas DataFrame. Provides a summary of:
- Total instances
- Class distribution
- Attribute statistics

- **Parameter**:
  - `filepath` (str): Path to the JSON dataset file.

---

### `save_model(filepath)`
Saves the trained model (priors and likelihoods) to a JSON file.

- **Parameter**:
  - `filepath` (str): Path to save the model.

---

### `load_model(filepath)`
Loads a pre-trained model from a JSON file.

- **Parameter**:
  - `filepath` (str): Path to the model file.

---

### `test()`
Predicts class labels for instances in the test dataset and evaluates the model's performance. 

- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - Specificity
  - F1-Score

Displays a confusion matrix and computed metrics.

---

## Usage

### 1. Prepare Datasets
- Ensure the datasets are in JSON format with a target column named `PlayTennis`.

Example JSON format:
```json
[
  {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
  {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"}
]
```

### 2. Train the Model
```python
model = Bayesian(train_dataset='train.json', test_dataset='test.json')
model.train()
```

### 3. Save the Model
```python
model.save_model('bayesian_model.json')
```

### 4. Load the Model
```python
model.load_model('bayesian_model.json')
```

### 5. Test the Model
```python
model.test()
```

---

## Example Workflow

```python
# Initialize the Bayesian classifier
model = Bayesian(train_dataset='train.json', test_dataset='test.json')

# Train the model
model.train()

# Save the trained model
model.save_model('model.json')

# Load the model
model.load_model('model.json')

# Test the model and evaluate performance
model.test()
```

---

## Notes
- Use Laplace smoothing to handle unseen feature values in the test dataset.
- Ensure the `PlayTennis` column is the target variable for classification.

