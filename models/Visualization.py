#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time


# ### BoxPlot

# In[2]:


def generate_boxplot(data, x_label='X-axis label', y_label='Y-axis label', title='', xticks_rotation=45, figsize=(10, 6)):
    """
    Generate a boxplot image.

    Parameters:
        data (list or pandas DataFrame): Data for the boxplot. If a list, each element should be a 1D array representing a group.
                                         If a DataFrame, each column will be treated as a separate group.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title for the boxplot.
        xticks_rotation (int, optional): Rotation angle for the x-axis labels in degrees. Default is 45.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Default is (10, 6).

    Returns:
        str: Path to the generated boxplot image file.
    """

    # If data is a DataFrame, select only numeric columns
    # if isinstance(data, pd.DataFrame):
    #     numeric_cols = data.select_dtypes(include=['number']).columns
    #     data = data[numeric_cols]

    # Select numeric columns for correlation matrix
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data = data[numeric_cols]
    
    if len(numeric_cols) == 0:
        print("No suitable columns found for boxplot.")
        return

    # Create a boxplot
    plt.figure(figsize=figsize)
    plt.boxplot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(ticks=range(1, len(data.columns) + 1), labels=data.columns, rotation=xticks_rotation)  # Set x-axis tick labels
    # plt.tight_layout()  # Adjust layout to improve label visibility
    plt.grid(True)

    # Save the plot as a temporary file
    temp_file = 'boxplot.png'
    plt.savefig(temp_file)
    # plt.show()
    plt.close()  # Close the plot to release memory

    return temp_file


# ### Correlation_Matrix

# In[3]:


def plot_correlation_matrix(df, method='pearson', annot=True, cmap='coolwarm', figsize=(10, 8), save_path="correlation_matrix.png"):
    """
    Plot the correlation matrix for a given DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame for which the correlation matrix is to be plotted.
        method (str, optional): Method used to calculate the correlation. Default is 'pearson'.
                                Other options are 'kendall' and 'spearman'.
        annot (bool, optional): Whether to annotate the correlation values on the plot. Default is True.
        cmap (str or colormap, optional): The colormap to be used for the plot. Default is 'coolwarm'.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Default is (10, 8).
        save_path (str, optional): The path where the correlation matrix plot should be saved.
                                   If not provided, the plot will not be saved.

    Returns:
        None
    """
    # Select numeric columns for correlation matrix
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) == 0:
        print("No suitable columns found for correlation matrix.")
        return
    
    # Compute the correlation matrix
    corr = df[numeric_cols].corr(method=method)
    
    # Plot the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    
    # Save the plot in the working directory
    if save_path:
        plt.savefig(os.path.join(os.getcwd(), save_path))
        print(f"Correlation Matrix plot saved in the working directory as: {save_path}")
    else:
        plt.savefig(save_path)
        plt.close()
        print(f"Correlation Matrix plot saved in the working directory as: correlation_matrix.png")
    return save_path 
    # plt.show()
    


# ### BAR Chart

# In[4]:


def create_bar_chart(categories, values, title, xlabel, ylabel, filename=None):
    """
    Function to create a bar chart and save it in the same working directory.

    Parameters:
        categories (list): List of categories or labels for the x-axis.
        values (list): List of values corresponding to each category.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str, optional): Name of the file to save the chart. If None, the chart will be saved with a default filename.

    Returns:
        None
    """
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    if filename is None:
        # Generate default filename using current timestamp
        # filename = f"bar_chart_{int(time.time())}.png"
        filename = f"bar_chart.png"
    
    plt.savefig(filename, bbox_inches='tight')  # Save the chart with tight bounding box
    plt.close()
    # plt.show()
    return filename


# In[5]:


# Example usage:
# categories = ['Category A', 'Category B', 'Category C', 'Category D']
# values = [25, 50, 30, 45]
# title = 'Example Bar Chart'
# xlabel = 'Categories'
# ylabel = 'Values'

# create_bar_chart(categories, values, title, xlabel, ylabel)


# In[6]:


def create_bar_chart_from_dataframe(dataframe, x_columns, y_columns, title, xlabel, ylabel, filename=None):
    """
    Function to create a bar chart from a DataFrame using multiple columns and save it in the same working directory.

    Parameters:
        dataframe (DataFrame): DataFrame containing the data.
        x_columns (list): List of column names for the x-axis (categories).
        y_columns (list): List of column names for the y-axis (values).
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str, optional): Name of the file to save the chart. If None, the chart will be saved with a default filename.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    
    # Set the width of the bars based on the number of y_columns
    bar_width = 0.35 / len(y_columns)
    index = np.arange(len(dataframe))

    for i, y_column in enumerate(y_columns):
        ax.bar(index + i * bar_width, dataframe[y_column], bar_width, label=y_column)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(index + bar_width * (len(y_columns) - 1) / 2)  # Adjust x-ticks
    ax.set_xticklabels(dataframe[x_columns].values.flatten())  # Extract values from DataFrame
    ax.legend()

    if filename is None:
        filename = f"bar_chart_{'_'.join(x_columns)}_vs_{'_'.join(y_columns)}.png"  # Generate default filename
    
    plt.savefig(filename, bbox_inches='tight')  # Save the chart with tight bounding box
    plt.close()
    # plt.show()
    return filename


# ### Scatter Plot

# In[59]:


def custom_scatterplot(df, x_col, y_col, hue_col=None, x_label='X-axis Label', y_label='Y-axis Label', title='Scatter Plot', hue_order=None, filename='scatter_plot.png'):
    """
    Generate a scatter plot from DataFrame columns and save it to the current directory.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - x_col (str): The column name for the x-axis.
    - y_col (str): The column name for the y-axis.
    - hue_col (str, optional): The column name for the hue (color grouping).
    - x_label (str, optional): Label for the x-axis.
    - y_label (str, optional): Label for the y-axis.
    - title (str, optional): Title for the plot.
    - hue_order (list, optional): Order for the hue values.
    - filename (str, optional): Name of the file to save the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    
    if hue_col:
        unique_hues = df[hue_col].unique()
        if hue_order:
            unique_hues = hue_order
        for hue in unique_hues:
            subset_df = df[df[hue_col] == hue]
            plt.scatter(subset_df[x_col], subset_df[y_col], label=hue)
        plt.legend(title=hue_col)
    else:
        plt.scatter(df[x_col], df[y_col])
    
    filename = 'custom-scatter-plot.png'
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)  # Save the plot to the current directory
    # plt.show()
    plt.close()
    
    return filename






# In[56]:


def scatterplot_each_column(df, target_column, hue_column=None, hue_order=None, save_dir='scatter_plots'):
    """
    Generate scatter plots for each column of the DataFrame against a target column, with an optional hue and hue order.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - target_column (str): The column name for the target axis (y-axis).
    - hue_column (str, optional): The column name for the hue (color grouping).
    - hue_order (list, optional): Order for the hue values.
    - save_dir (str, optional): Directory to save the plots.

    Returns:
    - List of filepaths
    """
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), save_dir)):
        os.makedirs(save_dir)
    
    files_list = []
    
    for column in df.columns:
        if column != target_column:
            plt.figure(figsize=(8, 6))
            if hue_column:
                unique_hues = df[hue_column].unique()
                if hue_order:
                    unique_hues = hue_order
                for hue in unique_hues:
                    subset_df = df[df[hue_column] == hue]
                    plt.scatter(subset_df[column].astype(str), subset_df[target_column], label=hue)
                plt.legend(title=hue_column)
            else:
                plt.scatter(df[column].astype(str), df[target_column])
            plt.xlabel(column)
            plt.ylabel(target_column)
            plt.title(f'Scatter Plot of {column} vs {target_column}')
            # Save plot with a unique filename
            filename = f'{save_dir}/scatter_plot_{column}_vs_{target_column}.png'
            files_list.append(filename)
            plt.savefig(filename)
            # plt.show()
            plt.close()
            
    return files_list



def plot_dataframe_description(df, figsize=(10, 6), bar_width=0.35, error_bars=True, save_filename=None):
    """
    Plot the description of a pandas DataFrame and save it in the current working directory.

    Parameters:
    - df: pandas DataFrame for which description is to be plotted.
    - figsize: Tuple specifying the figure size (width, height) in inches. Default is (10, 6).
    - bar_width: Width of the bars in the bar chart. Default is 0.35.
    - error_bars: Boolean indicating whether to include error bars representing standard deviation. Default is True.
    - save_filename: File name (including extension) to save the plot. If None, the plot will not be saved. Default is None.

    Returns:
    - None (displays the plot)
    """
    description = df.describe().transpose()
    categories = description.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    index = range(len(categories))

    mean_values = description['mean']
    std_values = description['std'] if error_bars else None

    ax.bar(index, mean_values, bar_width, label='Mean', yerr=std_values, capsize=5)

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Description of DataFrame')
    ax.set_xticks(index)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    save_filename  = 'df_desc.png'
    # Save the plot in the current working directory if save_filename is provided
    if save_filename:
        plt.savefig(os.path.join(os.getcwd(), save_filename))
        plt.close()

    return save_filename
    # plt.show()



# Plot the description of the DataFrame with error bars and save it
# plot_dataframe_description(df, error_bars=True, save_filename='description_plot.png')


# In[73]:


import pandas as pd
import matplotlib.pyplot as plt

def plot_statistics_summary(df):
    """
    Plot the summary statistics of a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame for which summary statistics are to be plotted.

    Returns:
    - None (displays the plot)
    """
    # Calculate summary statistics
    statistics_summary = df.describe().transpose()
    mode = df.mode().transpose().iloc[0]  # mode() returns DataFrame, we take the first row
    statistics_summary['mode'] = mode

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot bars for each statistic except for standard deviation
    statistics_summary[['mean', '50%', 'mode', 'min', 'max']].plot(kind='bar', ax=plt.gca())

    # Add labels and title
    plt.title('Summary Statistics (Excluding Standard Deviation)')
    plt.xlabel('Numerical Columns')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(['Mean', 'Median', 'Mode', 'Min', 'Max'])

    # Show plot
    plt.tight_layout()
    # plt.show()

    # Plot standard deviation separately
    plt.figure(figsize=(12, 6))
    statistics_summary['std'].plot(kind='bar', color='orange')
    plt.title('Standard Deviation')
    plt.xlabel('Numerical Columns')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    # plt.show()
    
    plt.tight_layout()
    
    filename = 'stats-summary.png'
    plt.savefig(filename)
    plt.close()
    
    return filename





