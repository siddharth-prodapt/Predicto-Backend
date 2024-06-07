#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import chardet
import requests
from io import StringIO
import copy
import io

# Function to read file with auto-detected encoding and create DataFrame
# As data is coming in TXT format so for Demo purpose I have hard Coded this...

# def read_api_to_dataframe(api_url, keep_header=True):
#     """
#     Fetch data from an API URL, detect its encoding, and create a DataFrame.

#     Parameters:
#     - api_url: str
#         The URL of the API from which to fetch data.

#     Returns:
#     - pandas DataFrame
#         A DataFrame containing the fetched data.
#     """
#     # Fetch data from the API
#     response = requests.get(api_url)
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")

#     # Detect encoding of fetched data
#     rawdata = response.content[:100000]  # Read the first 100KB of the response for encoding detection
#     result = chardet.detect(rawdata)
#     encoding = result['encoding']
#     print(f'Encoding: {encoding}')

#     # Read the fetched data with detected encoding and create a DataFrame
#     # if file_format == 'csv':
#     #     df = pd.read_csv(response.content.decode(encoding), encoding=encoding)
#     # elif file_format == 'json':
#     #     df = pd.read_json(response.content)
#     # else:
#     #     raise ValueError("Unsupported file format. Supported formats: 'csv', 'json'")
    
#     # Read the fetched data with detected encoding and create a DataFrame
#     decoded_content = response.content.decode(encoding)
#     # Split the text data by lines and create a list of dictionaries
#     data_list = [line.strip().split(',') for line in decoded_content.split('\n')]
#     if keep_header:
#         # Assume the first row contains headers
#         headers = data_list[0]
#         # Create a DataFrame using the remaining rows and headers
#         df = pd.DataFrame(data_list[1:], columns=headers)
#     else:
#         df = pd.DataFrame(data_list)
    
#     return df
# In[9]:


# def process_file_from_apis(file_path_api_url, file_type_api_url):
#     """
#     Fetches a file from a given API endpoint based on its path and type,
#     decodes the file content using the detected encoding,
#     and returns a pandas DataFrame representing the file data.

#     Parameters:
#     - file_path_api_url (str): The URL of the API endpoint that provides the file path.
#     - file_type_api_url (str): The URL of the API endpoint that provides the file type (e.g., 'csv', 'xls', 'xlsx').

#     Returns:
#     - pd.DataFrame or None: A pandas DataFrame containing the file data if successful, or None if an error occurs.
#     """
#     try:
#         # Make HTTP request to get the file path
#         file_path_response = requests.get(file_path_api_url)
#         if file_path_response.status_code != 200:
#             raise Exception(f"Failed to retrieve file path. Status code: {file_path_response.status_code}")
        
#         # Extract file path from response
#         file_path = file_path_response.json().get('file_path')
#         if not file_path:
#             raise Exception("File path not found in API response")
        
#         # Make HTTP request to get the file type
#         file_type_response = requests.get(file_type_api_url)
#         if file_type_response.status_code != 200:
#             raise Exception(f"Failed to retrieve file type. Status code: {file_type_response.status_code}")
        
#         # Extract file type from response
#         file_type = file_type_response.json().get('file_type')
#         if not file_type:
#             raise Exception("File type not found in API response")
        
#         # Make HTTP request to fetch the file content
#         file_content_response = requests.get(file_path)
#         if file_content_response.status_code != 200:
#             raise Exception(f"Failed to retrieve file content. Status code: {file_content_response.status_code}")
        
#         # Decode the file content using chardet to detect encoding
#         file_content = file_content_response.content
#         encoding_result = chardet.detect(file_content)
#         file_encoding = encoding_result['encoding']
        
#         # Determine file format based on file type
#         if file_type.lower() == 'csv':
#             # Read CSV content into DataFrame
#             df = pd.read_csv(io.StringIO(file_content.decode(file_encoding)))
#         elif file_type.lower() in ['xls', 'xlsx']:
#             # Read Excel content into DataFrame
#             df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
#         else:
#             raise Exception(f"Unsupported file type: {file_type}")
        
#         return df
    
#     except Exception as e:
#         print(f"Error processing file: {e}")
#         return None

# # Example usage:
# # file_path_api_url = 'https://example.com/api/get_file_path'
# # file_type_api_url = 'https://example.com/api/get_file_type'

# result_df = process_file_from_apis(file_path_api_url, file_type_api_url)


# In[11]:


def process_file_from_apis(file_path_api_url, file_type):
    """
    Fetches a file from a given API endpoint based on its path,
    decodes the file content using the detected encoding,
    and returns a pandas DataFrame representing the file data.

    Parameters:
    - file_path_api_url (str): The URL of the API endpoint that provides the file path.
    - file_type (str): The type of the file ('csv', 'xls', 'xlsx').

    Returns:
    - pd.DataFrame or None: A pandas DataFrame containing the file data if successful, or None if an error occurs.
    """
    try:
        # Make HTTP request to get the file path
        file_path_response = requests.get(file_path_api_url)

        if file_path_response.status_code != 200:
            raise Exception(f"Failed to retrieve file path. Status code: {file_path_response.status_code}")
        # Extract file path from response
        file_path = file_path_response.json().get('file_path')
        print('File_Path', file_path)
        if not file_path:
            raise Exception("File path not found in API response")
        
        print("\n\n\n Hello zee \n\n\n")
        
        # Make HTTP request to fetch the file content
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)  # Read the first 100,000 bytes of the file

    # Use chardet to detect the encoding of the raw data
        result = chardet.detect(raw_data)

    # Return the detected encoding
        file_encoding =  result['encoding']
        
        # Determine file format based on file type
        if file_type.lower() == 'csv':
            # Read CSV content into DataFrame
            df = pd.read_csv(file_path, encoding= file_encoding)
        elif file_type.lower() in ['xls', 'xlsx']:
            # Read Excel content into DataFrame
            df = pd.read_excel(file_path, file_encoding)
        else:
            raise Exception(f"Unsupported file type: {file_type}")
        
        return df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


# In[5]:


def load_data_from_api(file_path_api_url, file_type_api_url):
    """
    Load file data from specified APIs into a pandas DataFrame.

    Parameters:
    - file_path_api_url (str): The URL of the API endpoint that provides the file path.
    - file_type_api_url (str): The URL of the API endpoint that provides the file type (e.g., 'csv', 'xls', 'xlsx').

    Returns:
    - pd.DataFrame or None: A pandas DataFrame containing the file data if successful, otherwise None.
    """
    try:
        # Call process_file_from_apis function to fetch and process file data
        df = process_file_from_apis(file_path_api_url, file_type_api_url)
        return df
    except Exception as e:
        print(f"Error loading data from API: {e}")
        return None


# In[2]:


def read_api_to_dataframe(api_url, keep_header=True):
    """
    Fetch data from an API URL, detect its encoding, and create a DataFrame.

    Parameters:
    - api_url: str
        The URL of the API from which to fetch data.
    - keep_header: bool, optional
        Whether to keep the first row as headers. Default is True.

    Returns:
    - pandas DataFrame
        A DataFrame containing the fetched data.
    """
    # Fetch data from the API
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")

    # Detect encoding of fetched data
    rawdata = response.content[:100000]  # Read the first 100KB of the response for encoding detection
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    print(f'Encoding: {encoding}')

    # Read the fetched data with detected encoding
    decoded_content = response.content.decode(encoding)

    # Split the decoded content into lines
    lines = decoded_content.split('\n')
    
    # Extract header and data rows
    header = lines[0].strip().split(',')
    data_rows = [row.strip().split(',') for row in lines[1:] if row.strip()]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header)
    
    # Convert columns to appropriate data types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Leave non-numeric columns as object type
    
    return df


# Example usage:
# df = read_api_to_dataframe('your_api_url_here')


# In[3]:


def load_data_from_url(url, keep_header=True):
    """
    Load data from a given URL into a DataFrame.

    Parameters:
    - url: str
        The URL from which to fetch the data.
    - keep_header: bool, optional (default=True)
        Whether to keep the first row as header. If False, all rows are treated as data.

    Returns:
    - pandas DataFrame or None
        A DataFrame containing the fetched data if successful, otherwise None.
    """
    
    try:
        # Load data from the API into a DataFrame
        df = read_api_to_dataframe(url, keep_header=keep_header)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


# In[4]:


def get_head(df, n=5):
    """
    Return the first n rows of the DataFrame as a list of dictionaries in JSON format.

    Parameters:
    - df: pandas DataFrame
        The DataFrame from which the first n rows will be retrieved.
    - n: int, optional (default=5)
        Number of rows to retrieve from the beginning of the DataFrame.

    Returns:
    - list of dictionaries
        A list of dictionaries representing the first n rows of the DataFrame.
    """
    
    head_df = df.head(n)
    head_json = head_df.to_json(orient='records')
    return json.loads(head_json)


# In[5]:


def get_tail(df, n=5):
    """
    Return the last n rows of the DataFrame as a list of dictionaries in JSON format.

    Parameters:
    - df: pandas DataFrame
        The DataFrame from which the last n rows will be retrieved.
    - n: int, optional (default=5)
        Number of rows to retrieve from the end of the DataFrame.

    Returns:
    - list of dictionaries
        A list of dictionaries representing the last n rows of the DataFrame.
    """
    
    tail_df = df.tail(n)
    tail_json = tail_df.to_json(orient='records')
    return json.loads(tail_json)


# In[6]:


def get_data_types(dataframe):
    """
    Get the data types of columns in a DataFrame and return as JSON.
    
    Args:
    dataframe (pandas.DataFrame): The DataFrame to get data types from.
    
    Returns:
    str: JSON string containing data types.
    """
    data_types = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
    return json.dumps(data_types)


# In[7]:


def get_dataframe_shape(dataframe):
    """
    Get the shape (number of rows and columns) of a DataFrame.
    
    Args:
    dataframe (pandas.DataFrame): The DataFrame to get the shape of.
    
    Returns:
    tuple: A tuple containing the number of rows and columns in the DataFrame.
    """
    return dataframe.shape


# In[8]:


def get_dataframe_size(dataframe):
    """
    Get the size (total number of elements) of a DataFrame.
    
    Args:
    dataframe (pandas.DataFrame): The DataFrame to get the size of.
    
    Returns:
    int: The total number of elements (rows * columns) in the DataFrame.
    """
    return dataframe.size


# In[9]:


def get_total_null_values(dataframe):
    """
    Get the total number of null values in each column of a DataFrame and return as JSON.
    
    Args:
    dataframe (pandas.DataFrame): The DataFrame to get the null values from.
    
    Returns:
    str: JSON string containing the total number of null values in each column.
    """
    null_values = dataframe.isnull().sum().to_dict()
    return json.dumps(null_values)


# In[10]:


# describe all the columns of the data - non numuric column also included
def describe_the_data(df):
    """ 9790913160
    Generate descriptive statistics for all columns of the DataFrame, including non-numeric columns.

    Parameters:
    - df: pandas DataFrame
        The DataFrame for which descriptive statistics will be generated.

    Returns:
    - str
        A JSON string containing descriptive statistics for all columns of the DataFrame.
    """
    description = df.describe(include='all').fillna(np.nan).to_dict()
    json_output = {}
    for col in description:
        json_output[col] = description[col]
    return json.dumps(json_output)


# In[11]:


# describe the columns of the data
def describe_dataframe(df):
    """
    Generate descriptive statistics for numeric columns of the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame for which descriptive statistics will be generated.

    Returns:
    - dict
        A dictionary containing descriptive statistics for numeric columns of the DataFrame.
    """
    description = df.describe().to_dict()
    return description


# In[12]:


# To get column_names in order
def get_column_names_in_order(df):
    """
    Get column names of a DataFrame along with their respective indices in order.

    Parameters:
    - df: pandas DataFrame
        The DataFrame whose column names are to be retrieved.

    Returns:
    - str
        A JSON string containing column names along with their respective indices.
    """
    column_names = df.columns.tolist()
    column_indices = {index + 1 : column_name for index, column_name in enumerate(column_names)}
    return json.dumps(column_indices)


# In[13]:


def get_dataframe_info(df):
    """
    Get information summary of the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame for which information summary will be retrieved.

    Returns:
    - str
        A JSON string containing the information summary of the DataFrame.
    """
    
    # Capture the summary information in a string
    string_buffer = StringIO()
    df.info(buf=string_buffer)
    info_summary = string_buffer.getvalue()
    string_buffer.close()
    
    # Create a dictionary with the summary information
    info_dict = {
        "info_summary": info_summary
    }
    
    # Convert the dictionary to JSON
    return json.dumps(info_dict)


# In[14]:


def drop_column(df, column_name):
    """
    Drops the specified column from the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame from which the column will be dropped.
    - column_name: str
        The name of the column to drop.

    Returns:
    - pandas DataFrame
        The DataFrame with the specified column dropped.
    """
    return df.drop(columns=[column_name])


# In[15]:


def drop_multiple_columns(df, columns_to_drop):
    """
    Drops multiple columns from a pandas DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame from which columns are to be dropped.
        columns_to_drop (list): A list of column names to be dropped from the DataFrame.
        
    Returns:
        DataFrame: A new DataFrame with the specified columns dropped.
    """
    return df.drop(columns=columns_to_drop)


# In[16]:


def interchange_columns(df, column1, column2):
    """
    Interchanges the positions of two columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame in which columns will be interchanged.
    - column1: str
        The name of the first column.
    - column2: str
        The name of the second column.

    Returns:
    - pandas DataFrame
        The DataFrame with the positions of the specified columns interchanged.
    """
    # Get a list of all column names
    columns = df.columns.tolist()

    # Find the indices of the columns
    index1 = columns.index(column1)
    index2 = columns.index(column2)

    # Swap the column positions in the list
    columns[index1], columns[index2] = columns[index2], columns[index1]

    # Reorder the DataFrame columns
    df = df[columns]

    return df


# In[17]:


def drag_column(df, from_position, to_position):
    """
    Drag a column from one position to another in the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame in which the column will be dragged.
    - from_position: int
        The current position of the column (1-indexed).
    - to_position: int
        The desired position to which the column will be dragged (1-indexed).

    Returns:
    - pandas DataFrame
        The DataFrame with the specified column dragged to the new position.
    """
    # Ensure positions are valid
    if from_position < 1 or from_position > len(df.columns) or to_position < 1 or to_position > len(df.columns):
        raise ValueError("Invalid positions provided.")

    # Convert positions to 0-indexed
    from_position -= 1
    to_position -= 1

    # Get column names
    columns = df.columns.tolist()

    # Remove the column from its current position
    column_to_drag = columns.pop(from_position)

    # Insert the column at the new position
    columns.insert(to_position, column_to_drag)

    # Reorder the DataFrame columns
    df = df[columns]

    return df



# In[18]:


def add_calculated_column(df, new_column_name, calculation):
    """
    Add a new column to the DataFrame by performing calculations based on existing columns.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to which the new column will be added.
    - new_column_name: str
        The name of the new column.
    - calculation: str
        The calculation expression based on existing columns. For example, 'column1 + column2'.

    Returns:
    - pandas DataFrame
        The DataFrame with the new column added.
    """
    # Evaluate the calculation expression within the context of the DataFrame
    df[new_column_name] = pd.eval(calculation, engine='python')

    return df


# In[19]:


def add_new_row(df, new_row_data):
    """
    Add a new row to a pandas DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame to which the new row will be added.
        new_row_data (dict): A dictionary where keys are column names and values are corresponding values for the new row.
        
    Returns:
        DataFrame: The DataFrame with the new row added.
    """
    # Convert the new row data to a DataFrame
    new_row_df = pd.DataFrame([new_row_data])
    
    # Append the new row to the original DataFrame
    df = df.append(new_row_df, ignore_index=True)
    
    return df


# In[20]:


def copy_dataframe(df):
    """
    Make a copy of the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be copied.

    Returns:
    - pandas DataFrame
        A copy of the original DataFrame.
    """
    return df.copy()


# In[21]:


def deep_copy_dataframe(df):
    """
    Make a deep copy of a pandas DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to be copied.

    Returns:
    DataFrame: A deep copy of the input DataFrame.
    """
    return copy.deepcopy(df)


# In[22]:


def save_dataframe(df, file_path, file_name):
    """
    Saves a pandas DataFrame to a specified location with a specified name.
    
    Parameters:
        df (DataFrame): The pandas DataFrame to be saved.
        file_path (str): The directory path where the file will be saved.
        file_name (str): The name of the file (without extension).
        
    Returns:
        str: The full path of the saved file.
    """
    # Construct the full file path
    full_file_path = f"{file_path}/{file_name}.csv"
    
    # Save the DataFrame to CSV
    df.to_csv(full_file_path, index=False)
    
    return full_file_path


# In[23]:


def convert_numeric_columns(df):
    """
    Convert columns containing numeric values as strings to numeric data type.
    
    Parameters:
        df (DataFrame): The pandas DataFrame.
        
    Returns:
        DataFrame: The DataFrame with numeric columns converted to the appropriate data type.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                # Attempt to convert the column to numeric
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                # If conversion fails, continue to the next column
                continue
    return df


# In[24]:


def convert_column_to_numeric(df, column_name):
    """
    Convert a specific column containing numeric values as strings to numeric data type.
    
    Parameters:
        df (DataFrame): The pandas DataFrame.
        column_name (str): The name of the column to be converted.
        
    Returns:
        DataFrame: The DataFrame with the specified column converted to the appropriate data type.
    """
    try:
        # Attempt to convert the specified column to numeric
        df[column_name] = pd.to_numeric(df[column_name])
    except ValueError:
        print(f"Conversion failed for column '{column_name}'.")
    return df


# # Test code After this...

# In[6]:


# Example usage:
# url = "http://10.169.60.84:8000/user/14/file?filename=ML-MATT-CompetitionQT1920_train_2.csv"

# url = 'http://10.169.60.84:8000/testfile'
# url = "https://4e11-161-69-80-128.ngrok-free.app/14/file/download?filename=ML-MATT-CompetitionQT1920_test.csv"
# df = load_data_from_url(url,keep_header=False)
# import pandas as pd
# import json
# filePath = "C:/Users/zeeshan.p/ML_Workspace/Anomaly Detection/Data/ML-MATT-CompetitionQT1920_test.csv"
# df = pd.read_csv(filePath, encoding='ISO-8859-1') 

# if df is not None:
#     print(df.head())  # Display the first few rows of the loaded DataFrame
# else:
#     print("Failed to load data.")
# print(df)


# import pandas as pd
# df = pd.read_csv('../Data/hiring.csv')


# In[7]:


# df.info()
# df.isna().sum()
# df.head()
# df.shape


# In[96]:


# df.isna().sum()


# In[97]:


# df.isnull().sum()


# In[98]:


# get_total_null_values(df)


# In[35]:


# describe_the_data(df)


# In[36]:


# get_dataframe_info(df)


# In[37]:


# df.describe()


# In[38]:


# describe_dataframe(df)


# In[25]:


# url = 'http://10.169.60.84:8000/user/14/file?filename=ML-MATT-CompetitionQT1920_test.csv'
# a = load_data_from_url(url, keep_header=True)


# In[26]:


# get_dataframe_info(a)


# In[99]:


# a.info()
# col1 = r'test_score(out of 10)'
# col2 = r'interview_score(out of 10)'


# In[100]:


# df = add_calculated_column(df, 'new_column_name', 'col1+col2')


# In[101]:


# df


# In[8]:


# Convert Notebook into Script
# !jupyter nbconvert --to script DataSetup.ipynb


# In[ ]:




