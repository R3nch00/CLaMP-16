# Classification of Malware(CLaMP-16)

This repository contains my thesis work on malware classification using the **CLaMP** dataset from Kaggle. The project was implemented in **Python**, executed through **Jupyter Notebooks**, and includes two versions of the notebook that document the progression of the thesis project:
        🔹 **CLaMP16_malware_classification_v1.ipynb** — Initial Version
        🔹 **CLaMP16_malware_classification.ipynb** — Updated/Final Version

## 📁 Repository Structure

CLaMP-16/
├── notebooks/
│ ├── CLaMP16_malware_classification_v1.ipynb # Initial version
│ └── CLaMP16_malware_classification.ipynb # Updated/final version
├── data/ # (Optional) Kaggle dataset files
├── requirements.txt # Python dependencies
└── README.md # This file


## 📝 Notebooks Overview

### 📘 `CLaMP16_malware_classification_v1.ipynb`


### 📙 `CLaMP16_malware_classification_v2.ipynb`


## 🛠️ Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/R3nch00/CLaMP-16.git
   cd CLaMP-16

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **[Optional] Download the CLaMP dataset**
   •	Source: Kaggle (“CLaMP – Malware Classification”)
   •	Save CSV files under data/

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/

5. **Open notebooks in detail**
   •	v1.ipynb: Original pipeline and baseline
   •	.ipynb: Final, polished modeling steps

## 📊 Methodology
Malware classification pipeline was developed through a structured and iterative approach involving data preprocessing, model experimentation, and performance evaluation. The key stages of the methodology are as follows
  🧹 **Data Loading & Cleaning** - Loaded and cleaned CLaMP dataset: handled missing values; label-encoded categories; normalized features
  📊 **Exploratory Data Analysis (EDA)** - Visualized class distributions, feature correlations, and stats; identified imbalances to guide preprocessing
  🛠️ **Feature Engineering** - Selected key features via correlation and importance; applied dimensionality reduction; prepared data for ML and DL models
  🤖 **Model Training** - The project involves extensive experimentation with both machine learning and deep learning models for malware classification using the CLaMP dataset
        🔹 Traditional Machine Learning Models - These models were trained with varying feature sets and evaluated using standard classification metrics (accuracy, precision, recall, F1-score, ROC-AUC).
              K-Nearest Neighbors (KNN)
              Random Forest (RF)
              Naive Bayes (NB)
              AdaBoost
              Logistic Regression (LR)
              Decision Tree (DT)
              Linear Discriminant Analysis (LDA)
``
        🔹 Deep Learning Models - Deep learning models were implemented using TensorFlow/Keras and trained on processed numerical features. Hyperparameter tuning and architectural variations were applied to optimize performance.
              Multilayer Perceptron (MLP)
              1D Convolutional Neural Network (1D-CNN)
              Recurrent Neural Network (RNN)   
              Long Short-Term Memory (LSTM)
              Gated Recurrent Unit (GRU)
  📈 **Evaluation Metrics** - Recorded, compared, and analyzed model performance to identify top approaches for CLaMP malware classification
        •	Accuracy
        •	Precision, Recall, and F1-Score
        •	Confusion Matrix
        •	ROC-AUC Curve

## ✅ Dependencies
This project was developed in Python using common data science libraries. To run the notebooks, make sure the following packages are installed -
   ```bash
   numpy
   pandas
   scikit-learn
   xgboost
   matplotlib
   seaborn
   jupyter
```
## 🔮 Future Work
   •	Integrate deep learning (e.g., CNN or transformer models)
   •	Build a REST API endpoint for real time malware prediction
   •	Visualize predictions using interactive dashboards (e.g., Plotly, Dash)



🧪 Methodology
The malware classification pipeline was developed through a structured and iterative approach involving data preprocessing, model experimentation, and performance evaluation. The key stages are:

📥 Data Loading & Cleaning

Loaded and cleaned the CLaMP dataset

Handled missing values

Label-encoded categorical variables

Normalized numerical features

📊 Exploratory Data Analysis (EDA)

Visualized class distributions and feature correlations

Identified imbalances to guide preprocessing

🛠️ Feature Engineering

Selected key features based on correlation and importance

Applied dimensionality reduction techniques

Prepared datasets for ML and DL models
