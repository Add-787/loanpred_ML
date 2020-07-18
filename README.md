# loanpred_ML

Assignment during Online Internship with DLithe www.dlithe.com

Project Statement
The idea behind this project is to build a model that will classify whether a loan will be granted to a customer or not. It is based on the user’s marital status, education, number of dependents,employments, credit history,loan amount, gender etc.

# Approach
I have used logistic regression to predict the loan status of the applicant.

# Algorithm used
Logistic regression is basically a supervised classification algorithm. In a classification problem, the target variable(or output), y, can take only discrete values for given set of features(or inputs), X.



The Program

# Importing the libraries
import pandas as pd
import numpy as np

# Displaying dataset
df_train=pd.read_csv('train.csv')
df_train

# Find the no.of missing values
total = df_train.isnull().sum().sort_values(ascending=False)
total


# Imputing Missing values with mean for continuous variable
df_train['LoanAmount'].fillna(df_train['LoanAmount'].mean(), inplace=True)
df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mean(), inplace=True)
df_train['ApplicantIncome'].fillna(df_train['ApplicantIncome'].mean(), inplace=True)
df_train['CoapplicantIncome'].fillna(df_train['CoapplicantIncome'].mean(), inplace=True)

# Imputing Missing values with mode for categorical variables
df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True)
df_train['Married'].fillna(df_train['Married'].mode()[0], inplace=True)
df_train['Dependents'].fillna(df_train['Dependents'].mode()[0], inplace=True)
df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mode()[0], inplace=True)
df_train['Credit_History'].fillna(df_train['Credit_History'].mode()[0], inplace=True)


# Replace missing value of Self_Employed with more frequent category
df_train['Self_Employed'].fillna('No',inplace=True)

#Importing seaborn library for graph visualizations
import seaborn as sns
%matplotlib inline

#Analysing the independent variables with the target variables
sns.countplot(y='Gender',hue='Loan_Status',data=df_train)

sns.countplot(y='Married',hue='Loan_Status',data=df_train)


#Import label encoder to encode columns containing strings with numerical values.
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']

for var in cat:
    le = preprocessing.LabelEncoder()
    df_train[var]=le.fit_transform(df_train[var].astype('str'))
df_train.dtypes

#Importing Libraries to build the logistic regression model.
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

#Obtaining the training and test variables via train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Building the model
model = LogisticRegression()
model.fit(X_train, y_train)

#Used to get the accuracy of our model, our logistic regression model gives around 89% accuracy. As shown in the python notebook.
ypred = model.predict(X_test)
print(ypred)

























