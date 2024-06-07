#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus

from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os
from sklearn.metrics import silhouette_samples


# In[2]:


def kmeans_clustering(dataframe, k):
    """
    Perform k-means clustering on the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.

    Returns:
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    """

    # Extract numerical data from the dataframe
    X = dataframe.values

    # Initialize k-means algorithm
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(X)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids


# In[3]:


def kmeans_random_initialization(dataframe, k):
    """
    Perform k-means clustering with random initialization.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.

    Returns:
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    """
    X = dataframe.values
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


# In[4]:


def kmeans_kmeans_plusplus_initialization(dataframe, k):
    """
    Perform k-means clustering with k-means++ initialization.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.

    Returns:
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    """
    X = dataframe.values
    kmeans = KMeans(n_clusters=k, init=kmeans_plusplus)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


# In[5]:


def plot_elbow_method(dataframe, max_k, filename="elbow_method_plot.png"):
    """
    Plot the elbow method to determine the optimal number of clusters.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    max_k (int): The maximum number of clusters to consider.
    filename (str): Name of the file to save the plot (default: 'elbow_method_plot.png').
    """
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataframe)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(filename)
    plt.close()


# In[6]:


def silhouette_score_analysis(dataframe, max_k, filename="silhouette_score_plot.png"):
    """
    Analyze silhouette scores to determine the optimal number of clusters.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    max_k (int): The maximum number of clusters to consider.
    filename (str): Name of the file to save the plot (default: 'silhouette_score_plot.png').
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataframe)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(dataframe, labels))
    plt.plot(range(2, max_k + 1), silhouette_scores)
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(filename)
    plt.close()


# In[7]:


def mini_batch_kmeans_clustering(dataframe, k, batch_size=100):
    """
    Perform mini-batch k-means clustering on the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.
    batch_size (int): The number of samples to use for each batch.

    Returns:
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    """
    X = dataframe.values
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


# In[8]:


def online_learning_kmeans(dataframe, k):
    """
    Perform online learning with k-means on the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.

    Returns:
    kmeans (sklearn.cluster.KMeans): Trained k-means model.
    """
    X = dataframe.values
    kmeans = KMeans(n_clusters=k)
    for sample in X:
        kmeans.partial_fit([sample])
    return kmeans


# In[9]:


def predict_cluster_labels(dataframe, kmeans_model):
    """
    Predict cluster labels for new data points using a trained k-means model.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the new data points.
    kmeans_model (sklearn.cluster.KMeans): Trained k-means model.

    Returns:
    labels (numpy.ndarray): Array containing predicted cluster labels for each new data point.
    """
    X_new = dataframe.values
    return kmeans_model.predict(X_new)


# In[24]:


def silhouette_plot(dataframe, labels, filename="silhouette_plot.png"):
    """
    Plot silhouette scores for each sample to visualize clustering quality.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    filename (str): Name of the file to save the plot (default: 'silhouette_plot.png').
    """

    silhouette_vals = silhouette_samples(dataframe, labels)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(np.bincount(labels)):
        y_ax_upper += c
        color = plt.cm.viridis(float(i) / len(np.bincount(labels)))
        plt.barh(range(y_ax_lower, y_ax_upper), silhouette_vals[labels == i], height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += c
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, range(len(np.bincount(labels))))
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.savefig(filename)
    plt.close()
    print(f"Silhouette plot saved as '{filename}' in the current working directory.")



# In[25]:


def visualize_cluster_centroids(dataframe, centroids, filename="cluster_centroids_plot.png"):
    """
    Visualize cluster centroids in the feature space.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    filename (str): Name of the file to save the plot (default: 'cluster_centroids_plot.png').
    """

    df_centroids = pd.DataFrame(centroids, columns=dataframe.columns)
    sns.scatterplot(data=dataframe, alpha=0.5)
    sns.scatterplot(data=df_centroids, marker='x', color='red', s=100)
    plt.title('Cluster Centroids Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Data Points', 'Centroids'])
    plt.savefig(filename)
    plt.close()
    print(f"Cluster centroids plot saved as '{filename}' in the current working directory.")




# In[12]:


def compute_davies_bouldin_index(dataframe, labels):
    """
    Compute the Davies-Bouldin index to evaluate clustering performance.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.

    Returns:
    db_index (float): Davies-Bouldin index.
    """
    return davies_bouldin_score(dataframe, labels)


# In[13]:


def compute_centroid_distances(centroids):
    """
    Compute the distance between each pair of cluster centroids.

    Parameters:
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.

    Returns:
    centroid_distances (numpy.ndarray): Array containing distances between each pair of centroids.
    """
    centroid_distances = cdist(centroids, centroids)
    return centroid_distances


# In[29]:


def plot_cluster_size_distribution(labels, filename="cluster_size_distribution_plot.png"):
    """
    Visualize the distribution of cluster sizes.

    Parameters:
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    filename (str): Name of the file to save the plot (default: 'cluster_size_distribution_plot.png').
    """

    sns.histplot(labels, bins=len(np.unique(labels)), discrete=True)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Data Points')
    plt.savefig(filename)
    plt.close()
    print(f"Cluster size distribution plot saved as '{filename}' in the current working directory.")



# In[30]:


def detect_cluster_outliers(dataframe, labels, centroids, threshold=2.0):
    """
    Identify potential outliers in each cluster based on distance from the centroid.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    centroids (numpy.ndarray): Array containing coordinates of cluster centroids.
    threshold (float): Threshold for outlier detection (default: 2.0).

    Returns:
    outlier_indices (list): List of indices corresponding to potential outliers.
    """
    outlier_indices = []
    for cluster_label in np.unique(labels):
        cluster_data = dataframe[labels == cluster_label]
        centroid = centroids[cluster_label]
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        cluster_outliers = cluster_data[distances > threshold]
        outlier_indices.extend(cluster_outliers.index.tolist())
    return outlier_indices


# In[31]:


def cluster_membership_analysis(dataframe, labels):
    """
    Analyze the membership of each data point in its assigned cluster.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.

    Returns:
    membership_df (pandas.DataFrame): DataFrame containing data point indices and their corresponding cluster labels.
    """
    membership_df = pd.DataFrame({'Data Point Index': range(len(dataframe)), 'Cluster Label': labels})
    return membership_df


# In[32]:


def cluster_stability_analysis(dataframe, k, num_runs=10):
    """
    Evaluate the stability of clusters across multiple runs of k-means with different random initializations.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    k (int): The number of clusters.
    num_runs (int): Number of runs of k-means with different initializations (default: 10).

    Returns:
    silhouette_scores (list): List of silhouette scores for each run.
    """
    silhouette_scores = []
    for _ in range(num_runs):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataframe)
        silhouette_scores.append(silhouette_score(dataframe, kmeans.labels_))
    return silhouette_scores


# In[33]:


def visualize_clusters(dataframe, labels, method='pca', n_components=2, plot_title='Cluster Visualization', filename="cluster_visualization_plot.png"):
    """
    Visualize the clusters in a lower-dimensional space using dimensionality reduction techniques.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    method (str): Dimensionality reduction method ('pca' or 'tsne', default: 'pca').
    n_components (int): Number of components for dimensionality reduction (default: 2).
    plot_title (str): Title for the plot (default: 'Cluster Visualization').
    filename (str): Name of the file to save the plot (default: 'cluster_visualization_plot.png').
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    reduced_data = reducer.fit_transform(dataframe)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title(plot_title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Cluster')
    plt.savefig(filename)
    plt.close()
    print(f"Cluster visualization plot saved as '{filename}' in the current working directory.")


# In[34]:


def cluster_profiling(dataframe, labels):
    """
    Analyze the characteristics of each cluster by computing statistics for each feature within the cluster.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.

    Returns:
    cluster_profiles (pandas.DataFrame): DataFrame containing statistics for each feature within each cluster.
    """
    cluster_profiles = dataframe.groupby(labels).agg(['mean', 'median', 'std'])
    return cluster_profiles


# In[35]:


def compare_clusters(dataframe, labels1, labels2):
    """
    Compare the clusters obtained from two different sets of labels.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels1 (numpy.ndarray): Array containing cluster labels for each data point (first set).
    labels2 (numpy.ndarray): Array containing cluster labels for each data point (second set).

    Returns:
    cluster_comparison_df (pandas.DataFrame): DataFrame containing comparison of clusters.
    """
    cluster_comparison_df = pd.DataFrame({'Labels Set 1': labels1, 'Labels Set 2': labels2})
    return cluster_comparison_df


# In[36]:


def interpret_clusters(dataframe, labels, feature_importance=None):
    """
    Interpret the meaning of clusters by identifying important features contributing to each cluster.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the dataset.
    labels (numpy.ndarray): Array containing cluster labels for each data point.
    feature_importance (dict): Dictionary containing feature importance scores (optional).

    Returns:
    cluster_interpretation (dict): Dictionary containing interpreted clusters.
    """
    cluster_interpretation = {}
    for cluster_label in np.unique(labels):
        cluster_data = dataframe[labels == cluster_label]
        if feature_importance:
            important_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            important_features = [feature for feature, _ in important_features]
        else:
            important_features = cluster_data.mean().sort_values(ascending=False).index.tolist()
        cluster_interpretation[cluster_label] = important_features
    return cluster_interpretation


# In[37]:


def cluster_label_consistency(labels_list):
    """
    Evaluate the consistency of cluster labels across multiple sets of labels.

    Parameters:
    labels_list (list): List of arrays containing cluster labels for each data point (multiple sets).

    Returns:
    label_consistency_scores (list): List of consistency scores for each pair of label sets.
    """
    from sklearn.metrics import adjusted_rand_score

    label_consistency_scores = []
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            consistency_score = adjusted_rand_score(labels_list[i], labels_list[j])
            label_consistency_scores.append(consistency_score)
    return label_consistency_scores


# In[ ]:





# In[38]:


# Convert Notebook into Script
# get_ipython().system('jupyter nbconvert --to script Kmeans.ipynb')


# In[ ]:




