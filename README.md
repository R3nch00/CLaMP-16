# Classification of Malware(CLaMP-16)

This repository contains my thesis work on malware classification using the **CLaMP** dataset from Kaggle. The project was implemented in **Python**, executed through **Jupyter Notebooks**, and includes two versions of the notebook that document the progression of the thesis project:
         🔹 **CLaMP16_malware_classification_v1.ipynb** — Initial Version
         🔹 **CLaMP16_malware_classification.ipynb** — Updated/Final Version

## 📁 Repository Structure
```bash
CLaMP-16/
├── notebooks/
│ ├── CLaMP16_malware_classification_v1.ipynb # Initial version
│ └── CLaMP16_malware_classification.ipynb # Updated/final version
├── data/ # (Optional) Kaggle dataset files
├── requirements.txt # Python dependencies
└── README.md # This file
```

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

## 🧪 Methodology
Malware classification pipeline was developed through a structured and iterative approach involving data preprocessing, model experimentation, and performance evaluation. The key stages of the methodology are as follows -
1. 📥 **Data Loading & Cleaning**
         •	Loaded and cleaned the CLaMP dataset
         •	Handled missing values
         •	Label-encoded categorical variables
         •	Normalized numerical features
          
2. 📊 **Exploratory Data Analysis (EDA)**
        •	Visualized class distributions and feature correlations
        •	Identified imbalances to guide preprocessing

3. 🛠️ **Feature Engineering**
        •	Selected key features based on correlation and importance
        •	Applied dimensionality reduction techniques
        •	Prepared datasets for ML and DL models
        
4. 🤖 **Model Training**
        **Traditional ML Models**: KNN, RF, NB, AdaBoost, LR, DT, LDA
        •	Trained with various feature sets
        •	Evaluated using accuracy, precision, recall, F1-score, ROC-AUC
        **Deep Learning Models**: MLP, 1D-CNN, RNN, LSTM, GRU
        •	Built with TensorFlow/Keras
        •	Applied hyperparameter tuning and architectural variations

5. 📈 **Evaluation Metrics**
        Compared model performance using:
```bash
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC Curve
```
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




