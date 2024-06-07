#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor


# ### Split Data

# In[7]:


def split_data(df, target_columns):
    """
    Splits the dataframe into X (features) and y (target variables).

    Parameters:
    - df: pandas DataFrame, the dataframe to be split
    - target_columns: list of str, the names of the target columns

    Returns:
    - X: pandas DataFrame, contains the features
    - y: pandas DataFrame, contains the target variables
    """
    X = df.drop(columns=target_columns)  # Exclude the target columns to get features
    Y = df[target_columns]  # Get the target columns
    return X, Y


# In[8]:


def train_test_split_data(df, target_columns, test_size=0.2, random_state=None):
    """
    Splits the dataframe into train and test sets.

    Parameters:
    - df: pandas DataFrame, the dataframe to be split
    - target_columns: list of str, the names of the target columns
    - test_size: float, optional, default=0.2, the proportion of the dataset to include in the test split
    - random_state: int, optional, default=None, controls the random seed for reproducibility

    Returns:
    - X_train: pandas DataFrame, features for training
    - X_test: pandas DataFrame, features for testing
    - y_train: pandas DataFrame, target variables for training
    - y_test: pandas DataFrame, target variables for testing
    """
    X = df.drop(columns=target_columns)  # Features
    y = df[target_columns]  # Target variables

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# ### Linear Reg
def perform_linear_regression(data_frame, x, y):
    # Extracting x and y data
    # x = data_frame[[x_column]]
    # y = data_frame[y_column]

    # Creating and fitting the linear regression model
    # model = LinearRegression()
    model = LogisticRegression()
    model.fit(x, y)

    return model
def perform_linear_regression_with_time_estimation(data_frame, x_column, y_column):
    start_time = time.time()

    print('start_time :- ',start_time)

    # Perform linear regression
    regression_model = perform_linear_regression(data_frame, x_column, y_column)

    # Calculate time taken
    end_time = time.time()
    print('end_time :- ',end_time)
    time_taken = end_time - start_time
    print('time_taken :- ',time_taken)

    return regression_model, time_taken

# ### Machine Learning Model Selection

# In[9]:


def create_model(model_type):
    """
    Creates an instance of the specified machine learning model.

    Parameters:
        model_type: A string specifying the type of machine learning model.
                    Allowed values: 'linear_regression', 'logistic_regression',
                                    'decision_tree_classifier', 'decision_tree_regressor',
                                    'random_forest_classifier', 'random_forest_regressor',
                                    'svc', 'svr', 'kneighbors_classifier', 'kneighbors_regressor',
                                    'gradient_boosting_classifier', 'gradient_boosting_regressor',
                                    'gaussian_nb', 'mlp_classifier', 'mlp_regressor'.

    Returns:
        Instance of the specified machine learning model.
    """
    if model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'logistic_regression':
        return LogisticRegression()
    elif model_type == 'decision_tree_classifier':
        return DecisionTreeClassifier()
    elif model_type == 'decision_tree_regressor':
        return DecisionTreeRegressor()
    elif model_type == 'random_forest_classifier':
        return RandomForestClassifier(n_estimators=15)
    elif model_type == 'random_forest_regressor':
        return RandomForestRegressor()
    elif model_type == 'svc':
        return SVC()
    elif model_type == 'svr':
        return SVR()
    elif model_type == 'kneighbors_classifier':
        return KNeighborsClassifier()
    elif model_type == 'kneighbors_regressor':
        return KNeighborsRegressor()
    elif model_type == 'gradient_boosting_classifier':
        return GradientBoostingClassifier()
    elif model_type == 'gradient_boosting_regressor':
        return GradientBoostingRegressor()
    elif model_type == 'gaussian_nb':
        return GaussianNB()
    elif model_type == 'mlp_classifier':
        return MLPClassifier()
    elif model_type == 'mlp_regressor':
        return MLPRegressor()
    else:
        raise ValueError("Invalid model type. Please choose from the specified list.")


# ### Training of Model

# In[10]:


def fit_model(model, X_train, y_train):
    """
    Fits a machine learning model to the provided training data.

    Parameters:
        model: The machine learning model object (e.g., from scikit-learn).
        X_train: The training data features (independent variables).
        y_train: The training data labels (dependent variable).

    Returns:
        Fitted model.
    """
    try:
        # Fit the model to the training data
        model.fit(X_train, y_train)
        print("Model fitted successfully!")
        return model
    except Exception as e:
        print("An error occurred while fitting the model:", str(e))
        return None


# ### Predictions using Trained Model

# In[11]:


def predict_with_model(model, X):
    """
    Predicts target variables using a pre-trained model.

    Parameters:
    - model: trained model object
    - X: pandas DataFrame or numpy array, features for prediction

    Returns:
    - predictions: numpy array, predicted target variables
    """
    predictions = model.predict(X)
    return predictions


# ### Performance Matrics

# In[12]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(y_true, y_pred, task='classification'):
    """
    Evaluates the predictions by calculating appropriate metrics based on the task.

    Parameters:
    - y_true: pandas Series or numpy array, true target values
    - y_pred: pandas Series or numpy array, predicted target values
    - task: str, optional, default='classification', specify the task ('classification' or 'regression')

    Returns:
    - metrics: dict, containing evaluation metrics
    """
    metrics = {}

    # if isinstance(y_true, (int, float)):
    #     y_true = np.array([y_true])

    # if isinstance(y_pred, (int, float)):
    #     y_pred = np.array([y_pred])

    # if y_true.dtype != y_pred.dtype:
    #     raise ValueError("Mismatched data types between y_true and y_pred.")


    if task == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
    elif task == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'Mean Squared Error (MSE)': mse,
            'Mean Absolute Error (MAE)': mae,
            'R-squared (R^2)': r2
        }
    else:
        raise ValueError("Invalid task type. Please specify 'classification' or 'regression'.")

    return metrics


# In[40]:


import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix for a classification task.

    Parameters:
    - y_true: array-like
        True labels.
    - y_pred: array-like
        Predicted labels.
    - classes: list
        List of class labels.

    Returns:
    - None
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()
    plt.savefig('confusion_matrix.png')  # Save the plot as an image
    plt.close()


# ### Save Model

# In[14]:


import joblib

def save_model(model, file_path, file_name):
    """
    Save a trained machine learning model to a specified file path with a given filename.

    Parameters:
    - model: object
        The trained machine learning model to be saved.
    - file_path: str
        The directory path where the model will be saved.
    - file_name: str
        The filename for the saved model (without extension).

    Returns:
    - str
        The full path of the saved model.
    """
    # Construct the full file path
    full_file_path = f"{file_path}/{file_name}.joblib"
    
    # Save the model to the specified file path
    joblib.dump(model, full_file_path)
    
    return full_file_path


# In[15]:


import pickle

def save_model_as_pickle(model, file_path, file_name):
    """
    Save a trained machine learning model to a specified file path with a given filename in .pkl format.

    Parameters:
    - model: object
        The trained machine learning model to be saved.
    - file_path: str
        The directory path where the model will be saved.
    - file_name: str
        The filename for the saved model (without extension).

    Returns:
    - str
        The full path of the saved model.
    """
    # Construct the full file path
    full_file_path = f"{file_path}/{file_name}.pkl"
    
    # Save the model to the specified file path
    with open(full_file_path, 'wb') as file:
        pickle.dump(model, file)
    
    return full_file_path


# ### Test

# In[17]:


# Assuming filePath contains the path of the file obtained from the API
# filePath = "C:/Users/zeeshan.p/ML_Workspace/Anomaly Detection/Data/ML-MATT-CompetitionQT1920_train.csv"
# df = pd.read_csv(filePath, encoding='ISO-8859-1') 


# In[24]:


# df.head()


# In[25]:


# df = df.drop(columns=['Time','CellName','maxUE_UL+DL'])


# In[26]:


# target_columns = ['Unusual']


# In[27]:


# X_train, X_test, y_train, y_test = train_test_split_data(df, target_columns, test_size=0.2, random_state=None)

# Regression Models:

# Linear Regression (linear_regression)
# Decision Tree Regressor (decision_tree_regressor)
# Random Forest Regressor (random_forest_regressor)
# SVR (Support Vector Regressor) (svr)
# KNeighbors Regressor (kneighbors_regressor)
# Gradient Boosting Regressor (gradient_boosting_regressor)

# Classification Models:

# Logistic Regression (logistic_regression)
# Decision Tree Classifier (decision_tree_classifier)
# Random Forest Classifier (random_forest_classifier)
# SVC (Support Vector Classifier) (svc)
# KNeighbors Classifier (kneighbors_classifier)
# In[28]:


# model = create_model('logistic_regression')
# model = create_model('decision_tree_classifier')
# model = create_model('random_forest_classifier')
# model = create_model('svc')
# model = create_model('kneighbors_classifier')
# model = create_model('gradient_boosting_classifier')
# model = create_model('mlp_classifier')
# model = create_model('gaussian_nb')


# In[29]:


# model = create_model('linear_regression')
# model = create_model('decision_tree_regressor')
# model = create_model('random_forest_regressor')
# model = create_model('svr')
# model = create_model('kneighbors_regressor')
# model = create_model('gradient_boosting_regressor')
# model = create_model('mlp_regressor')


# In[30]:


# Trained_model = fit_model(model, X_train, y_train)


# In[31]:


# pri_value = predict_with_model(Trained_model, X_test)


# In[13]:


# evaluate_predictions(d, z)
# evaluate_predictions(y_test,pri_value, task='regression')
# evaluate_predictions(y_test,pri_value)


# In[41]:


# classes = [0,1]
# plot_confusion_matrix(y_test, pri_value, classes)


# In[15]:


# Convert Notebook into Script
# !jupyter nbconvert --to script Models.ipynb


# In[ ]:




