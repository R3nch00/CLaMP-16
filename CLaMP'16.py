import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dabl
import random
import plotly.express as px
import math
import optuna
import imblearn
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
df = pd.read_csv('C:/Users/O M A R/Desktop/kaggle/ClaMP_Integrated-5184.csv')
print(df)
print(df.head())
print(df.info())
print(df.columns)
print(df.shape)
print(df.describe())
numeric_df = df.select_dtypes(include=['number'])
null_values=df.isnull().sum()
print(null_values)

corr = numeric_df.corr()

# Create a heatmap of the correlation matrix
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
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")
    # Display the plot
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

    # Customize the plot
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

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

    #Test Train Split fro modeling purpose
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


#k-nearest neighbors(KNN) Classifications

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_arr,y_arr)
sklearn_preds = knn.predict(X_test_arr)

score = roc_auc_score(y_test_arr, sklearn_preds)
print('1. ROC AUC: %.3f' % score)
print('2. Accuracy :',accuracy_score(y_test_arr, sklearn_preds))
print('3. Classification Report -\n',classification_report(y_test_arr, sklearn_preds))
print('4. Confusion Matrix - \n',confusion_matrix(y_test_arr, sklearn_preds))


#Random Forest(RF) Classifier with default hyperparameters

rf_clf=RandomForestClassifier(random_state=100,n_jobs=-1)
rf_clf.fit(X_train,y_train)
rf_pred=rf_clf.predict(X_test)
rf_pred_proba=rf_clf.predict_proba(X_test)

score=roc_auc_score(y_test,rf_pred)
print('1. ROC AUC: %.3f' % score)
print('2. Accuracy :',accuracy_score(y_test, rf_pred))
print('3. Classification Report -\n',classification_report(y_test, rf_pred))
print('4. Confusion Matrix - \n',confusion_matrix(y_test, rf_pred))


#Naive Bayes CLassifier(NB)

df=df.drop(['NumberOfSections','CreationYear'],axis=1)
le=LabelEncoder()
for i in df:
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])
    else:
        continue


X=df.drop(['class'],axis=1)
y=df['class']

X_train,X_test,y_train,y_test=train_test_split(X,y)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred[:15])

accuracy=accuracy_score(y_pred,y_test)
print(accuracy)
print(classification_report(y_test,y_pred))
print('F1 Score: ',f1_score(y_test,y_pred,zero_division=1))
confmat=confusion_matrix(y_true=y_test,y_pred=y_pred)


print(confmat)

fig, ax =plt.subplots(figsize=(12.5, 12.5))
ax.matshow(confmat,  cmap=plt.cm.Blues, alpha=0.30)
for i in range(confmat.shape[0]):
  for j in range(confmat.shape[1]):
    ax.text(x=j, y=i,
            s=confmat[i, j],
            va='center', ha='center')
    plt.title('Using NBClassifier model at 90% accuracy prediction & 91% F-1 score on the malware dataset for identify false negatives and false positives as well as true positives and true negatives with 0 benine and 1 had a malware')
    plt.xlabel('Predicted label')
    plt.ylabel('True Label')
    
