#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


# ### Handling null values

# In[1]:


def drop_null_values(df):
    """
    Drops rows with null values from a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame from which to drop null values.
    
    Returns:
    pandas.DataFrame: DataFrame with null values dropped.
    """
    cleaned_df = df.dropna()
    return cleaned_df


# In[2]:


def drop_null_values_column(df, column_name):
    """
    Drops rows with null values from a specific column of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame from which to drop null values.
    column_name (str): The name of the column from which to drop null values.
    
    Returns:
    pandas.DataFrame: DataFrame with null values in the specified column dropped.
    """
    cleaned_df = df.dropna(subset=[column_name])
    return cleaned_df


# In[15]:


def fill_null_with_stats(df, column_name, strategy='mean'):
    """
    Fills null values in a specific column of a DataFrame with mean, median, or mode.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame in which to fill null values.
    column_name (str): The name of the column in which to fill null values.
    strategy (str): The strategy to use for filling null values. Options are 'mean', 'median', or 'mode'.
                    Default is 'mean'.
    
    Returns:
    None. The DataFrame is modified in place.
    """
    if strategy == 'mean':
        fill_value = df[column_name].mean()
    elif strategy == 'median':
        fill_value = df[column_name].median()
    elif strategy == 'mode':
        fill_value = df[column_name].mode()[0]
    else:
        raise ValueError("Invalid strategy. Please choose from 'mean', 'median', or 'mode'.")

    df[column_name] = df[column_name].fillna(fill_value)


# In[92]:


def fill_missing_values(data, method='ffill'):
    """
    Fill missing values in a DataFrame using forward fill (ffill) or backward fill (bfill).

    Parameters:
    - data: pandas DataFrame, the input DataFrame with missing values.
    - method: str, either 'ffill' for forward fill or 'bfill' for backward fill. Default is 'ffill'.

    Returns:
    - filled_data: pandas DataFrame, DataFrame with missing values filled using the specified method.
    """
    if method not in ['ffill', 'bfill']:
        raise ValueError("Invalid method. Use 'ffill' for forward fill or 'bfill' for backward fill.")

    if method == 'ffill':
        filled_data = data.ffill()
    else:
        filled_data = data.bfill()

    return filled_data


# In[102]:


def fill_column_missing_values(data, column_name, method='ffill'):
    """
    Fill missing values in a specific column of a DataFrame using forward fill (ffill) or backward fill (bfill).

    Parameters:
    - data: pandas DataFrame, the input DataFrame.
    - column_name: str, the name of the column containing missing values.
    - method: str, either 'ffill' for forward fill or 'bfill' for backward fill. Default is 'ffill'.

    Returns:
    - filled_data: pandas DataFrame, DataFrame with missing values in the specified column filled using the specified method.
    """
    if method not in ['ffill', 'bfill']:
        raise ValueError("Invalid method. Use 'ffill' for forward fill or 'bfill' for backward fill.")

    if method == 'ffill':
        filled_data = data.copy()
        filled_data[column_name] = filled_data[column_name].ffill()
    else:
        filled_data = data.copy()
        filled_data[column_name] = filled_data[column_name].bfill()

    return filled_data


# In[107]:


def linear_interpolation(data):
    """
    Perform linear interpolation for missing values in a DataFrame.

    Parameters:
    - data: pandas DataFrame, the input DataFrame with missing values.

    Returns:
    - interpolated_data: pandas DataFrame, DataFrame with missing values interpolated using linear interpolation.
    """
    interpolated_data = data.interpolate(method='linear', axis=0, limit_direction='both')

    return interpolated_data


# In[111]:


def linear_interpolation_columns(data, columns):
    """
    Perform linear interpolation for missing values in specified columns of a DataFrame.

    Parameters:
    - data: pandas DataFrame, the input DataFrame with missing values.
    - columns: list of str, the names of the columns to perform linear interpolation.

    Returns:
    - interpolated_data: pandas DataFrame, DataFrame with missing values in specified columns interpolated using linear interpolation.
    """
    interpolated_data = data.copy()
    for column in columns:
        interpolated_data[column] = interpolated_data[column].interpolate(method='linear', limit_direction='both')

    return interpolated_data


# In[113]:


def knn_imputation(data, n_neighbors=5):
    """
    Perform K-Nearest Neighbors (KNN) imputation for missing values in a DataFrame.

    Parameters:
    - data: pandas DataFrame, the input DataFrame with missing values.
    - n_neighbors: int, the number of nearest neighbors to use for imputation. Default is 5.

    Returns:
    - imputed_data: pandas DataFrame, DataFrame with missing values imputed using KNN imputation.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return imputed_data


# In[118]:


def knn_imputation_columns(data, columns, n_neighbors=5):
    """
    Perform K-Nearest Neighbors (KNN) imputation for missing values in specified columns of a DataFrame.

    Parameters:
    - data: pandas DataFrame, the input DataFrame with missing values.
    - columns: list of str, the names of the columns to perform KNN imputation on.
    - n_neighbors: int, the number of nearest neighbors to use for imputation. Default is 5.

    Returns:
    - imputed_data: pandas DataFrame, DataFrame with missing values in specified columns imputed using KNN imputation.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = data.copy()
    imputed_data[columns] = imputer.fit_transform(data[columns])

    return imputed_data


# In[119]:


def fill_null_with_attribute(df, column_name, chosen_attribute):
    """
    Fill the null values of a column with a specific attribute chosen by the user.

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to fill null values.
    chosen_attribute: The attribute chosen by the user to fill null values.

    Returns:
    None. The DataFrame is modified in place.
    """
    df[column_name] = df[column_name].fillna(chosen_attribute)



# ### Unique values for categorical field

# In[33]:


def count_unique_elements(df):
    """
    Get the number of unique elements in each column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    str: A JSON string containing the number of unique elements in each column.
    """
    unique_counts = df.nunique()
    unique_counts_json = unique_counts.to_json()
    return unique_counts_json


# In[34]:


def get_unique_elements(df, column_name):
    """
    Get all the unique elements of a specific column in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    column_name (str): The name of the column to extract unique elements from.

    Returns:
    list: A list containing all the unique elements of the specified column.
    """
    unique_elements = df[column_name].unique().tolist()
    return unique_elements


# ### Remove duplicate rows

# In[38]:


def delete_duplicate_rows(df):
    """
    Delete duplicate rows from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to remove duplicate rows from.

    Returns:
    pandas.DataFrame: DataFrame with duplicate rows removed.
    """
    cleaned_df = df.drop_duplicates()
    return cleaned_df


# In[59]:


# Check with different data source
def identify_datetime_columns(df):
    """
    Identify datetime columns and their subsets (date, time, datetime) in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    dict: A dictionary containing the names of datetime columns and their subsets.
    """
    datetime_columns = {}

    for column in df.columns:
        try:
            # Convert column to datetime if it contains only strings
            if df[column].dtype == 'object':
                converted_values = pd.to_datetime(df[column])
            else:
                continue
        except:
            continue
        
        # Check if all values are datetime
        if pd.api.types.is_datetime64_any_dtype(converted_values):
            datetime_columns[column] = "datetime"
        # elif pd.api.types.is_timedelta64_dtype(converted_values):
        #     datetime_columns[column] = "timedelta"
        # elif pd.api.types.is_datetime64_ns_dtype(converted_values):
        #     datetime_columns[column] = "datetime_ns"
        # elif pd.api.types.is_timedelta64_ns_dtype(converted_values):
        #     datetime_columns[column] = "timedelta_ns"
            
        # Check if all values are dates (without time)
        elif pd.api.types.is_datetime64_any_dtype(converted_values.dt.date):
            datetime_columns[column] = "date"
        # Check if all values are times (without date)
        elif pd.api.types.is_datetime64_any_dtype(converted_values.dt.time):
            datetime_columns[column] = "time"

    return datetime_columns


# ### Normalization

# In[67]:


def z_score_normalization(dataframe):
    """
    Perform Z-score normalization on the numeric columns of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be normalized.

    Returns:
    normalized_dataframe (pandas DataFrame): Z-score normalized DataFrame.
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    
    # Calculate mean and standard deviation for each numeric column
    mean = numeric_cols.mean()
    std_dev = numeric_cols.std()
    
    # Perform Z-score normalization on each numeric column
    normalized_dataframe = (numeric_cols - mean) / std_dev
    
    # Replace original numeric columns with normalized values
    for col in numeric_cols.columns:
        dataframe[col] = normalized_dataframe[col]
    
    return dataframe


# In[75]:


def z_score_normalization_column(dataframe, column_name):
    """
    Perform Z-score normalization on a specific column of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be normalized.
    column_name (str): Name of the column to be normalized.

    Returns:
    normalized_dataframe (pandas DataFrame): DataFrame with the specified column Z-score normalized.
    """
    # Calculate mean and standard deviation for the specified column
    mean = dataframe[column_name].mean()
    std_dev = dataframe[column_name].std()
    
    # Perform Z-score normalization on the specified column
    normalized_column = (dataframe[column_name] - mean) / std_dev
    
    # Update the specified column in the original DataFrame with the normalized values
    normalized_dataframe = dataframe.copy()
    normalized_dataframe[column_name] = normalized_column
    
    return normalized_dataframe


# In[76]:


def mean_normalization_dataframe(dataframe):
    """
    Perform mean normalization on the numeric columns of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be normalized.

    Returns:
    normalized_dataframe (pandas DataFrame): DataFrame with numeric columns mean normalized.
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    
    # Calculate mean for each numeric column
    means = numeric_cols.mean()
    
    # Perform mean normalization on each numeric column
    normalized_numeric_cols = (numeric_cols - means) / (numeric_cols.max() - numeric_cols.min())
    
    # Replace original numeric columns with mean normalized values
    for col in numeric_cols.columns:
        dataframe[col] = normalized_numeric_cols[col]
    
    return dataframe


# In[77]:


def mean_normalization_column(dataframe, column_name):
    """
    Perform mean normalization on a specific column of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be normalized.
    column_name (str): Name of the column to be normalized.

    Returns:
    normalized_dataframe (pandas DataFrame): DataFrame with the specified column mean normalized.
    """
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(dataframe[column_name]):
        # Calculate mean for the specified column
        mean = dataframe[column_name].mean()
        
        # Perform mean normalization on the specified column
        normalized_column = (dataframe[column_name] - mean) / (dataframe[column_name].max() - dataframe[column_name].min())
        
        # Update the specified column in the original DataFrame with the mean normalized values
        normalized_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
        normalized_dataframe[column_name] = normalized_column
    else:
        # If the column contains non-numeric data, just return the original DataFrame
        normalized_dataframe = dataframe
    
    return normalized_dataframe


# In[80]:


def min_max_scaling_dataframe(dataframe):
    """
    Perform feature scaling (min-max scaling) on the numeric columns of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with numeric columns feature scaled.
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    
    # Perform min-max scaling on each numeric column
    scaled_numeric_cols = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())
    
    # Replace original numeric columns with scaled values
    scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
    for col in numeric_cols.columns:
        scaled_dataframe[col] = scaled_numeric_cols[col]
    
    return scaled_dataframe


# In[81]:


def min_max_scaling_column(dataframe, column_name):
    """
    Perform feature scaling (min-max scaling) on a specific column of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.
    column_name (str): Name of the column to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with the specified column feature scaled.
    """
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(dataframe[column_name]):
        # Perform min-max scaling on the specified column
        min_val = dataframe[column_name].min()
        max_val = dataframe[column_name].max()
        scaled_column = (dataframe[column_name] - min_val) / (max_val - min_val)
        
        # Update the specified column in the original DataFrame with the scaled values
        scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
        scaled_dataframe[column_name] = scaled_column
    else:
        # If the column contains non-numeric data, just return the original DataFrame
        scaled_dataframe = dataframe
    
    return scaled_dataframe
    


# In[82]:


def robust_scaling_dataframe(dataframe):
    """
    Perform Robust Scaling on the numeric columns of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with numeric columns Robustly Scaled.
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    
    # Perform Robust Scaling on each numeric column
    scaled_numeric_cols = (numeric_cols - numeric_cols.median()) / (numeric_cols.quantile(0.75) - numeric_cols.quantile(0.25))
    
    # Replace original numeric columns with scaled values
    scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
    for col in numeric_cols.columns:
        scaled_dataframe[col] = scaled_numeric_cols[col]
    
    return scaled_dataframe


# In[83]:


def robust_scaling_column(dataframe, column_name):
    """
    Perform Robust Scaling on a specific column of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.
    column_name (str): Name of the column to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with the specified column Robustly Scaled.
    """
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(dataframe[column_name]):
        # Perform Robust Scaling on the specified column
        median = dataframe[column_name].median()
        q1 = dataframe[column_name].quantile(0.25)
        q3 = dataframe[column_name].quantile(0.75)
        iqr = q3 - q1
        
        scaled_column = (dataframe[column_name] - median) / iqr
        
        # Update the specified column in the original DataFrame with the scaled values
        scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
        scaled_dataframe[column_name] = scaled_column
    else:
        # If the column contains non-numeric data, just return the original DataFrame
        scaled_dataframe = dataframe
    
    return scaled_dataframe


# In[84]:


def unit_vector_scaling_dataframe(dataframe):
    """
    Perform Unit Vector Scaling (Vector Normalization) on the numeric columns of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with numeric columns Unit Vector Scaled.
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number])
    
    # Perform Unit Vector Scaling on each numeric column
    norms = np.linalg.norm(numeric_cols, axis=0)
    scaled_numeric_cols = numeric_cols.div(norms)
    
    # Replace original numeric columns with scaled values
    scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
    for col in numeric_cols.columns:
        scaled_dataframe[col] = scaled_numeric_cols[col]
    
    return scaled_dataframe


# In[85]:


def unit_vector_scaling_column(dataframe, column_name):
    """
    Perform Unit Vector Scaling (Vector Normalization) on a specific column of the input DataFrame.

    Parameters:
    dataframe (pandas DataFrame): Input DataFrame containing the data to be scaled.
    column_name (str): Name of the column to be scaled.

    Returns:
    scaled_dataframe (pandas DataFrame): DataFrame with the specified column Unit Vector Scaled.
    """
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(dataframe[column_name]):
        # Perform Unit Vector Scaling on the specified column
        norms = np.linalg.norm(dataframe[column_name])
        scaled_column = dataframe[column_name] / norms
        
        # Update the specified column in the original DataFrame with the scaled values
        scaled_dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
        scaled_dataframe[column_name] = scaled_column
    else:
        # If the column contains non-numeric data, just return the original DataFrame
        scaled_dataframe = dataframe
    
    return scaled_dataframe


# ### Lable Encoder

# In[8]:


def label_encode_column(df, column_name):
    """
    Apply LabelEncoder to encode categorical labels in a specific column of a DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame.
        column_name (str): The name of the column to be label encoded.
        
    Returns:
        DataFrame: The DataFrame with the specified column label encoded.
        LabelEncoder: The fitted LabelEncoder object for potential inverse transformation.
    """
    label_encoder = LabelEncoder()
    df[column_name] = label_encoder.fit_transform(df[column_name])
    return df, label_encoder


# In[10]:


def inverse_label_encode_column(df, column_name, label_encoder):
    """
    Apply inverse transformation to revert label encoded values to original categorical labels in a specific column of a DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame.
        column_name (str): The name of the column to be inverse transformed.
        label_encoder (LabelEncoder): The fitted LabelEncoder object used for label encoding.
        
    Returns:
        DataFrame: The DataFrame with the specified column inverse transformed.
    """
    df[column_name] = label_encoder.inverse_transform(df[column_name])
    return df


# # Test 

# In[12]:


# Assuming filePath contains the path of the file obtained from the API
# filePath = "C:/Users/zeeshan.p/ML_Workspace/Anomaly Detection/Data/ML-MATT-CompetitionQT1920_test.csv"
# df = pd.read_csv(filePath) 


# In[13]:


# Convert Notebook into Script
# get_ipython().system('jupyter nbconvert --to script PreProcessing.ipynb')


# In[11]:


# df.isnull().sum()


# In[5]:


# a = fill_missing_values(df, method='ffill')


# In[7]:


# a.isnull().sum()


# In[ ]:




