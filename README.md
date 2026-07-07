# Credit Card Fraud Detection using Autoencoders

A deep learning project that detects fraudulent credit card transactions using an unsupervised **Autoencoder Neural Network** built with Keras/TensorFlow. Instead of learning to classify fraud directly, the model learns what a *normal* transaction looks like — and flags anything it can't reconstruct well as a potential fraud.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [Choosing a Threshold](#choosing-a-threshold)
- [Limitations & Future Work](#limitations--future-work)

## Overview

Credit card fraud costs billions of dollars every year, and fraudulent transactions make up a tiny fraction of all transactions — which makes this a highly **imbalanced classification problem**. Traditional supervised classifiers often struggle here because there simply aren't enough fraud examples to learn from.

This project takes a different approach: an **Autoencoder** is trained only on legitimate (non-fraudulent) transactions. It learns to compress and reconstruct normal transaction patterns. When a new transaction comes in, the model tries to reconstruct it — if the transaction is normal, reconstruction error will be low; if it's fraudulent (a pattern the model has never seen), the reconstruction error will be high. Transactions with a reconstruction error above a chosen threshold are flagged as fraud.

## How It Works

1. **Train on normal data only** — the autoencoder never sees fraud examples during training.
2. **Reconstruct** each transaction by encoding it into a compressed representation and decoding it back.
3. **Measure reconstruction error** (Mean Squared Error) between the original and reconstructed transaction.
4. **Flag anomalies** — transactions with reconstruction error above a set threshold are classified as fraud.

This is effectively **anomaly detection**, not standard classification, which makes it well-suited to problems with very few positive (fraud) examples.

## Dataset

This project uses the popular **Credit Card Fraud Detection** dataset available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraudulent transactions | 492 (~0.17%) |
| Features | 30 (`Time`, `Amount`, and `V1`–`V28`) |
| Target | `Class` (0 = normal, 1 = fraud) |

Because of confidentiality, most features (`V1`–`V28`) have already been transformed via **PCA**. Only `Time` (seconds elapsed since the first transaction) and `Amount` (transaction amount) remain in their original form.

> **Note:** The dataset is **not included** in this repository due to size and licensing. Download `creditcard.csv` from Kaggle and place it in a `data/` folder as described below.

## Tech Stack

- **Python 3**
- **TensorFlow** 1.x / **Keras** 2.x — model building and training
- **scikit-learn** — train/test split, scaling, and evaluation metrics
- **Pandas** & **NumPy** — data loading and manipulation
- **Matplotlib** & **Seaborn** — data visualization and plotting
- **Jupyter Notebook** — interactive development and analysis

## Model Architecture

The autoencoder is a simple, fully-connected (dense) network with 4 layers:

| Layer | Units | Activation | Role |
|---|---|---|---|
| Input | 29 | – | Input transaction features |
| Encoder | 14 | tanh (L1 regularized) | Compress input |
| Encoder | 7 | relu | Bottleneck representation |
| Decoder | 7 | tanh | Begin reconstruction |
| Decoder | 29 | relu | Reconstructed output |

**Training configuration:**
- Loss: Mean Squared Error (reconstruction loss)
- Optimizer: Adam
- Epochs: 100
- Batch size: 32
- Only the best-performing model (lowest validation loss) is saved via `ModelCheckpoint`
- Training metrics are logged for **TensorBoard**

## Project Structure

```
├── fraud_detection.ipynb   # Main notebook: EDA, model training, evaluation
├── model.h5                # Saved/trained Keras autoencoder model
├── data/
│   └── creditcard.csv      # Dataset (not included — download separately)
├── LICENSE                 # MIT License
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Make sure you have Python 3 installed, along with the following packages:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn tensorflow keras jupyter
```

> This notebook was originally built with **TensorFlow 1.2** and **Keras 2.0.4**. It should also run on newer TensorFlow/Keras versions with minor adjustments (e.g. import paths), but for a first run, using compatible versions will save you troubleshooting time.

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shubhamraj22/Credit-Card-Fraud-Detection-using-AutoEncoder.git
   cd <Credit-Card-Fraud-Detection-using-AutoEncoder>
   ```

2. **Download the dataset**
   - Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Create a `data/` folder in the project root and place the CSV inside:
     ```
     data/creditcard.csv
     ```

3. **Launch the notebook**
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```

## Usage

Run the notebook cells in order to:

1. **Load and explore the data** — check class distribution, transaction amounts, and time patterns for normal vs. fraudulent transactions.
2. **Preprocess the data** — drop the `Time` column, scale `Amount` using `StandardScaler`, and split into train/test sets (the model trains only on normal transactions).
3. **Build and train the autoencoder** — defined and compiled in Keras, trained for 100 epochs.
4. **Evaluate the model** — reconstruction error distributions, ROC curve, and precision-recall curve.
5. **Set a threshold and predict** — classify transactions as fraud/normal based on reconstruction error, and inspect the confusion matrix.

To reuse the already-trained model without retraining:

```python
from keras.models import load_model

autoencoder = load_model('model.h5')
predictions = autoencoder.predict(X_test)
```

## Results & Evaluation

The notebook evaluates the model using multiple metrics suited for imbalanced data:

- **Reconstruction error distribution** for normal vs. fraudulent transactions
- **ROC Curve** and AUC score
- **Precision-Recall Curve** (more informative than ROC for highly imbalanced datasets)
- **Confusion Matrix** at a chosen classification threshold

## Choosing a Threshold

Since the model outputs a continuous reconstruction error rather than a class label, a **threshold** must be chosen to decide what counts as "fraud." The notebook uses an example threshold (`2.9`), but this value directly controls the trade-off between:

- **Higher threshold** → fewer false positives, but may miss some fraud (lower recall)
- **Lower threshold** → catches more fraud, but flags more normal transactions as suspicious (lower precision)

The right threshold depends on your specific tolerance for false positives vs. false negatives, and can be tuned using the precision-recall curve generated in the notebook.

## Limitations & Future Work

- The dataset's features are PCA-transformed, so real-world interpretability of individual features is limited.
- The current model uses a simple dense autoencoder; more advanced architectures (variational autoencoders, LSTM-based autoencoders for sequential transaction data, or ensemble methods) could improve performance.
- The classification threshold is currently set manually; this could be automated using techniques like statistical outlier detection or cost-sensitive optimization.
- Consider retraining and validating with more recent versions of TensorFlow/Keras for long-term maintainability.



⭐ If you found this project useful, consider giving it a star on GitHub!