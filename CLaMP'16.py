import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dabl
import numba
import random
import plotly.express as px
import math
import optuna
import imblearn
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,recall_score, accuracy_score, classification_report,f1_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc,roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('C:/Users/O M A R/Desktop/kaggle/ClaMP_Integrated-5184.csv')

#Display DataFrame Information
print(df)
print(df.head())
print(df.info())
print(df.columns)
print(df.shape)
print(df.describe())

#Check for 'null value'
numeric_df = df.select_dtypes(include=['number'])
null_values=df.isnull().sum()
print(null_values)


# Create a heatmap of the correlation matrix
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# Exploratory Data Analysis(EDA)
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr().stack().reset_index(name="correlation")

# Create the correlation plot
g = sns.relplot(
data=corr,
x="level_0", y="level_1", hue="correlation", size="correlation",
palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
height=10, sizes=(50, 250), size_norm=(-.2, .8))
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legend_handles:
    artist.set_edgecolor(".7")

    # Display the plot
print(df['packer_type'])
dabl.plot(df,target_col='class')
plt.show()

# Print the class distribution
class_distribution = df['class'].value_counts()
print("Class Distribution:")
print(class_distribution)
class_distribution.plot(kind='bar')
plt.title('Class Distribution (1=Malware, 0=Benign)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['Benign (0)', 'Malware (1)'], rotation=0)
plt.show()

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection: Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)  # Select top 5 features
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# Model Tuning: Grid Search with Cross-Validation
parameters = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv=5)
clf.fit(X_train_rfe, y_train)

# Cross-Validation: Evaluate model performance
scores = cross_val_score(clf, X_train_rfe, y_train, cv=5)
print("Cross-validation scores: ", scores)
print("Average cross-validation score: ", scores.mean())

# Make predictions on the test data
y_pred = clf.predict(X_test_rfe)
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the accuracy and confusion matrix
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Assuming 'conf_matrix' is predefined and contains your confusion matrix data
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=labels, fmt="", cmap='Blues')

fpr_poly, tpr_poly, poly_thresholds = roc_curve(y_test, y_pred)
roc_auc_poly = auc(fpr_poly, tpr_poly)

plt.figure()
plt.plot(fpr_poly, tpr_poly, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc_poly)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Polynomial Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

#SVC-Support Vector Classification
pt = df['packer_type'].unique()
p_types = {pt[i] : i for i in range(len(pt))}
temp = []
for t in df['packer_type']:
    temp.append(p_types[t])
df['pt_num'] = temp
cl = df.pop('class')
df.pop('packer_type') # packer_type column changed to pt_num column with corr. Integers

x_train, x_test, y_train, y_test = train_test_split(df, cl, random_state=0) #DataFrame was spillted into training and testing sets

  #Pipeline made to scale & classify data
pipeStd = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=0))])
pipeStd.fit(x_train, y_train)

  # Best parameters found SVM Classifier and cross-validated by using GridSe archCV & SVC
param_grid = {'svm__C':[0.1, 1, 10, 100, 200, 300], 'svm__gamma':[0.001, 0.005, 0.01, 0.1, 1]}

grid = GridSearchCV(pipeStd, param_grid, cv = 5, n_jobs = -1)
grid.fit(x_train.to_numpy(), y_train)

   # Some core classification metrics
print('SVC score after StdScaler: {:.3f}'.format(
    grid.score(x_test.to_numpy(), y_test)))
print("SVC's best score on cross validation: {:.3f}".format(
    grid.best_score_))
print("Classifier's best parameters: {}".format(grid.best_params_))
pred_val = grid.predict(x_test.to_numpy())
print(classification_report(
    y_test, pred_val, target_names=['benign', 'malicious'], digits=3))

  #ROC-AUC score with plot
fpr, tpr, thresholds = roc_curve(
    y_test, grid.best_estimator_['svm'].decision_function(
        grid.best_estimator_['scaler'].transform(
            x_test.to_numpy())))
auc = roc_auc_score(y_test, grid.best_estimator_['svm'].decision_function(
        grid.best_estimator_['scaler'].transform(
            x_test.to_numpy())))
close_zero = np.argmin(np.abs(thresholds))

plt.figure(figsize=(5, 5), dpi=200)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(auc),
    color='g')
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
    label="Absolute zero edge", fillstyle='none', color='r')
plt.legend(loc='lower right')
plt.title('SVC with StdScaler')

# Pre-processing the data to enable one-hot encoding on the categorical cloumns

type_df=pd.DataFrame(df.dtypes).reset_index()
type_df.columns=['cols', 'type']
type_df[type_df['type']=='object']['cols'].unique()

print('Total unique values in "packer_type":',df['packer_type'].nunique())

    #Extracting required levels only based on value counts
packer_unique_df=pd.DataFrame(df['packer_type'].value_counts()).reset_index()
packer_unique_df.columns=['packer_type','unique_count']
catg=packer_unique_df[packer_unique_df['unique_count']>10]['packer_type'].unique()

encoded=pd.get_dummies(df['packer_type'])
encoded=encoded[[col for col in list(encoded.columns) if col in catg]]
print('Shape of encode :',encoded.shape)

    #Concatenating the encoded columns
 #Conditional Automation
if set(catg).issubset(set(df.columns))==False:
    df=pd.concat([df,encoded],axis=1)
    df.drop(columns='packer_type',inplace=True)

df.shape


#Separate the target Col for Analysis & Scaling the data(Standard scaler)

    #Test Train Split for modeling purpose
X=df.loc[:,[cols for cols in df.columns if ('class' not in cols)]]
y=df.loc[:,[cols for cols in df.columns if 'class' in cols]]

    #Scaling the feature
scaler=StandardScaler()
X=scaler.fit_transform(X)

    #Splitting data into train-set
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=100)

print('Total Shape of Train X:',X_train.shape)
print('Total Shape of Train Y:',y_train.shape)
print('Total Shape of Test X:',X_test.shape)

X_arr=np.array(X_train)
X_test_arr=np.array(X_test)

y_arr=np.array(y_train).reshape(len(y_train),1)
y_test_arr=np.array(y_test).reshape(len(y_test),1)

print(X_arr.shape)
print(X_test_arr.shape)
print(y_arr.shape)


# k-nearest neighbors(KNN) Classifications

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_arr,y_arr)
sklearn_preds = knn.predict(X_test_arr)

# Calculate various metrics
roc_auc = roc_auc_score(y_test_arr, sklearn_preds)
accuracy = accuracy_score(y_test_arr, sklearn_preds)
classification_rep = classification_report(y_test_arr, sklearn_preds)
conf_matrix = confusion_matrix(y_test_arr, sklearn_preds)
f1 = f1_score(y_test_arr, sklearn_preds)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

#Random Forest(RF) Classifier with default hyperparameters

rf_clf = RandomForestClassifier(random_state=100, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_pred_proba = rf_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, rf_pred)
accuracy = accuracy_score(y_test, rf_pred)
classification_rep = classification_report(y_test, rf_pred)
conf_matrix = confusion_matrix(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

#Naive Bayes CLassifier(NB)

# Initialize the Naive Bayes classifier
nb_clf = GaussianNB()

# Fit the model
nb_clf.fit(X_train, y_train)

# Make predictions
nb_pred = nb_clf.predict(X_test)
nb_pred_proba = nb_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, nb_pred)
accuracy = accuracy_score(y_test, nb_pred)
classification_rep = classification_report(y_test, nb_pred)
conf_matrix = confusion_matrix(y_test, nb_pred)
f1 = f1_score(y_test, nb_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

# Adaboost Classifier
# Initialize the AdaBoost classifier
ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=100)

# Fit the model
ada_clf.fit(X_train, y_train)

# Make predictions
ada_pred = ada_clf.predict(X_test)
ada_pred_proba = ada_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, ada_pred)
accuracy = accuracy_score(y_test, ada_pred)
classification_rep = classification_report(y_test, ada_pred)
conf_matrix = confusion_matrix(y_test, ada_pred)
f1 = f1_score(y_test, ada_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

# Adaboost Model [Matrix]
confmat1=confusion_matrix(y_true=y_test, y_pred=y_clf)

print(confmat1)

fig, ax =plt.subplots(figsize=(12.5, 12.5))
ax.matshow(confmat1,  cmap=plt.cm.OrRd, alpha=0.30)
for i in range(confmat1.shape[0]):
  for j in range(confmat1.shape[1]):
    ax.text(x=j, y=i,
            s=confmat1[i, j],
            va='center', ha='center')
    plt.title('Using Adaboost model at 97% accuracy prediction & 97% F-1 score on the malware dataset for identify false negatives and false positives as well as true positives and true negatives. With 0 benine and 1 had a Malware.')
    plt.xlabel('Predicted label')
    plt.ylabel('True Label')
      
# Logistic Regression
# Initialize the Logistic Regression classifier
log_reg_clf = LogisticRegression(random_state=100, max_iter=1000)

# Fit the model
log_reg_clf.fit(X_train, y_train)

# Make predictions
log_reg_pred = log_reg_clf.predict(X_test)
log_reg_pred_proba = log_reg_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, log_reg_pred)
accuracy = accuracy_score(y_test, log_reg_pred)
classification_rep = classification_report(y_test, log_reg_pred)
conf_matrix = confusion_matrix(y_test, log_reg_pred)
f1 = f1_score(y_test, log_reg_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

# Decision Tree Classifier

# Initialize the Decision Tree classifier
dt_clf = DecisionTreeClassifier(random_state=100)

# Fit the model
dt_clf.fit(X_train, y_train)

# Make predictions
dt_pred = dt_clf.predict(X_test)
dt_pred_proba = dt_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, dt_pred)
accuracy = accuracy_score(y_test, dt_pred)
classification_rep = classification_report(y_test, dt_pred)
conf_matrix = confusion_matrix(y_test, dt_pred)
f1 = f1_score(y_test, dt_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

#Linear Discriminant Analysis (LDA)

# Initialize the Linear Discriminant Analysis classifier
lda_clf = LinearDiscriminantAnalysis()

# Fit the model
lda_clf.fit(X_train, y_train)

# Make predictions
lda_pred = lda_clf.predict(X_test)
lda_pred_proba = lda_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, lda_pred)
accuracy = accuracy_score(y_test, lda_pred)
classification_rep = classification_report(y_test, lda_pred)
conf_matrix = confusion_matrix(y_test, lda_pred)
f1 = f1_score(y_test, lda_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy :', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)


#### MODEL ###

# Multi-Layer Perceptron (MLP)

# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score

# Initialize the MLP classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)

# Fit the model
mlp_clf.fit(X_train, y_train)

# Make predictions
mlp_pred = mlp_clf.predict(X_test)
mlp_pred_proba = mlp_clf.predict_proba(X_test)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, mlp_pred)
accuracy = accuracy_score(y_test, mlp_pred)
classification_rep = classification_report(y_test, mlp_pred)
conf_matrix = confusion_matrix(y_test, mlp_pred)
f1 = f1_score(y_test, mlp_pred)

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy of MLP:', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

# 1D-CNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score

# Assuming X_train, X_test, y_train, y_test are already defined and preprocessed

# Reshape data for 1D-CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Initialize the 1D-CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
accuracy = accuracy_score(y_test_classes, y_pred_classes)
classification_rep = classification_report(y_test_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy of 1D- CNN:', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)


#Recurrent Neural Network

import numpy as np
import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score

# Assuming X_train, X_test, y_train, y_test are already defined and preprocessed

# Reshape data for RNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Initialize the RNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64))
model.add(LSTM(128))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
accuracy = accuracy_score(y_test_classes, y_pred_classes)
classification_rep = classification_report(y_test_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy of RNN:', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)

#Long-Short Term Memory(LSTM)

import numpy as np
import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score

# Assuming X_train, X_test, y_train, y_test are already defined and preprocessed

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Initialize the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=X_train.shape[1]))
model.add(LSTM(128))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate various metrics
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
accuracy = accuracy_score(y_test_classes, y_pred_classes)
classification_rep = classification_report(y_test_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

# Print the results
print('1. ROC AUC: %.3f' % roc_auc)
print('2. Accuracy of LSTM:', accuracy)
print('3. Classification Report -\n', classification_rep)
print('4. Confusion Matrix - \n', conf_matrix)
print('5. F1 Score: %.3f' % f1)



import pandas as pd

# Assuming your data is in a DataFrame called df
data = {
    'Algorithm': ['1D-CNN', 'LSTM', 'MLP', 'RNN', 'Random Forest', 'Decision Tree', 'KNN', 'SVM', 'Logistic Regression'],
    'Accuracy': [0.975, 0.975, 0.980, 0.975, 0.993, 0.977, 0.973, 0.945, 0.959],
    'Recall': [0.990, 0.990, 0.990, 0.990, 0.990, 0.980, 0.970, 0.940, 0.960],
    'F1 score': [0.975, 0.975, 0.981, 0.975, 0.993, 0.978, 0.750, 0.949, 0.962],
    'Precision': [0.960, 0.960, 0.980, 0.960, 0.990, 0.970, 0.980, 0.960, 0.970],
    'ROC AUC': [0.991, 0.991, 0.980, 0.991, 0.993, 0.977, 0.973, 0.945, 0.960]
}

df = pd.DataFrame(data)

# Calculate the average for each algorithm/model
df['Avg'] = df[['Accuracy', 'Recall', 'F1 score', 'Precision', 'ROC AUC']].mean(axis=1)

# Determine the best in how many metrics
best_counts = (df[['Accuracy', 'Recall', 'F1 score', 'Precision', 'ROC AUC']] == df[['Accuracy', 'Recall', 'F1 score', 'Precision', 'ROC AUC']].max()).sum(axis=1)
df['Best in how many matrix'] = best_counts

print(df)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming your data is in a DataFrame called df
data = {
    'Algorithm': ['1D-CNN', 'LSTM', 'MLP', 'RNN', 'Random Forest', 'Decision Tree', 'KNN', 'SVM', 'Logistic Regression'],
    'Accuracy': [0.975, 0.975, 0.980, 0.975, 0.993, 0.977, 0.973, 0.945, 0.959],
    'Recall': [0.990, 0.990, 0.990, 0.990, 0.990, 0.980, 0.970, 0.940, 0.960],
    'F1 score': [0.975, 0.975, 0.981, 0.975, 0.993, 0.978, 0.750, 0.949, 0.962],
    'Precision': [0.960, 0.960, 0.980, 0.960, 0.990, 0.970, 0.980, 0.960, 0.970],
    'ROC AUC': [0.991, 0.991, 0.980, 0.991, 0.993, 0.977, 0.973, 0.945, 0.960],
    'Avg': [0.9782, 0.9782, 0.9822, 0.9782, 0.9918, 0.9764, 0.9292, 0.9478, 0.9622],
    'Best in how many matrix': [1, 1, 1, 1, 5, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Normalize the metrics
scaler = MinMaxScaler()
metrics = ['Accuracy', 'Recall', 'F1 score', 'Precision', 'ROC AUC', 'Avg', 'Best in how many matrix']
df[metrics] = scaler.fit_transform(df[metrics])

# Calculate the composite score (equal weights for simplicity)
df['Composite Score'] = df[metrics].mean(axis=1)

# Rank the models based on the composite score
df['Rank'] = df['Composite Score'].rank(ascending=False, method='min')

# Sort the DataFrame by the rank
df_sorted = df.sort_values(by='Rank')

print("Ranking based on Composite Score:")
print(df_sorted[['Algorithm', 'Composite Score', 'Rank']])

