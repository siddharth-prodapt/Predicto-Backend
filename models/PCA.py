#!/usr/bin/env python
# coding: utf-8

# In[50]:


from sklearn.decomposition import PCA
# from datasets import load_dataset

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# In[33]:


def apply_pca(dataset, n_components):
    """
    Apply PCA (Principal Component Analysis) to a dataset.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - n_components: Number of components to keep (int)

    Returns:
    - Transformed dataset after PCA
    - PCA model
    """

    # Initialize PCA model
    pca = PCA(n_components=n_components)

    # Fit PCA model to the dataset and transform the data
    transformed_data = pca.fit_transform(dataset)

    return transformed_data, pca


# In[48]:


def apply_pca_Dataframe(dataset, n_components):
    """
    Apply PCA (Principal Component Analysis) to a dataset.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - n_components: Number of components to keep (int)

    Returns:
    - Transformed dataset after PCA (DataFrame)
    - PCA model
    """

    # Initialize PCA model
    pca = PCA(n_components=n_components)

    # Fit PCA model to the dataset and transform the data
    transformed_data = pca.fit_transform(dataset)

    # Convert the transformed data to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(n_components)])

    return transformed_df, pca


# In[34]:


def reconstruct_data(compressed_data, pca_model):
    """
    Reconstruct the original data from compressed data using PCA.

    Parameters:
    - compressed_data: Compressed data obtained after PCA (2D array-like)
    - pca_model: PCA model used for compression

    Returns:
    - Reconstructed original data
    """
    reconstructed_data = pca_model.inverse_transform(compressed_data)
    return reconstructed_data


# In[49]:


def reconstruct_data_dataframe(compressed_data, pca_model):
    """
    Reconstruct the original data from compressed data using PCA.

    Parameters:
    - compressed_data: Compressed data obtained after PCA (DataFrame)
    - pca_model: PCA model used for compression

    Returns:
    - Reconstructed original data (DataFrame)
    """
    reconstructed_data = pca_model.inverse_transform(compressed_data)
    reconstructed_df = pd.DataFrame(reconstructed_data, columns=compressed_data.columns)
    return reconstructed_df


# In[35]:


def compress_and_reconstruct(dataset, n_components):
    """
    Compress the dataset using PCA and then reconstruct it back to the original form.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - n_components: Number of components to keep (int)

    Returns:
    - Compressed data
    - Reconstructed original data
    """
    # Apply PCA
    compressed_data, pca_model = apply_pca(dataset, n_components)

    # Reconstruct original data
    reconstructed_data = reconstruct_data(compressed_data, pca_model)

    return compressed_data, reconstructed_data


# In[51]:


def compress_and_reconstruct_dataframe(dataset, n_components):
    """
    Compress the dataset using PCA and then reconstruct it back to the original form.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - n_components: Number of components to keep (int)

    Returns:
    - Compressed data (DataFrame)
    - Reconstructed original data (DataFrame)
    """
    # Apply PCA
    compressed_data, pca_model = apply_pca_Dataframe(dataset, n_components)

    # Reconstruct original data
    reconstructed_data = reconstruct_data_dataframe(compressed_data, pca_model)

    return compressed_data, reconstructed_data


# In[53]:


def explained_variance_ratio(pca_model):
    """
    Calculate explained variance ratio for each principal component.

    Parameters:
    - pca_model: PCA model

    Returns:
    - Explained variance ratio for each principal component
    """
    return pca_model.explained_variance_ratio_


# In[37]:


def choose_n_components(dataset, threshold=0.95):
    """
    Choose the number of components based on a threshold of explained variance.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - threshold: Threshold for cumulative explained variance (default: 0.95)

    Returns:
    - Optimal number of components
    """
    pca = PCA().fit(dataset)
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_explained_variance >= threshold) + 1
    return n_components


# In[54]:


def choose_n_components_dataframe(dataset, threshold=0.95):
    """
    Choose the number of components based on a threshold of explained variance.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - threshold: Threshold for cumulative explained variance (default: 0.95)

    Returns:
    - Optimal number of components
    """
    pca = PCA().fit(dataset)
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_explained_variance >= threshold) + 1
    return n_components


# In[38]:


def plot_scree(pca_model, filename="scree_plot.png"):
    """
    Plot scree plot showing the explained variance for each principal component.

    Parameters:
    - pca_model: PCA model
    - filename: Filename to save the plot (default: "scree_plot.png")

    Returns:
    - Filename of the saved plot
    """
    plt.plot(range(1, len(pca_model.explained_variance_) + 1), pca_model.explained_variance_, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.title('Scree Plot')
    plt.savefig(filename)
    plt.close()
    return filename


# In[39]:


def plot_cumulative_variance(pca_model, filename="cumulative_variance_plot.png"):
    """
    Plot cumulative explained variance.

    Parameters:
    - pca_model: PCA model
    - filename: Filename to save the plot (default: "cumulative_variance_plot.png")

    Returns:
    - Filename of the saved plot
    """
    plt.plot(range(1, len(pca_model.explained_variance_ratio_) + 1), np.cumsum(pca_model.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')
    plt.savefig(filename)
    plt.close()
    return filename


# In[40]:


def biplot(pca_model, data, labels=None, filename="biplot.png"):
    """
    Create biplot of PCA.

    Parameters:
    - pca_model: PCA model
    - data: Data to be plotted
    - labels: Labels for data points (optional)
    - filename: Filename to save the plot (default: "biplot.png")

    Returns:
    - Filename of the saved plot
    """
    if labels is not None:
        for label in set(labels):
            indices = labels == label
            plt.scatter(data[indices, 0], data[indices, 1], label=label)
    else:
        plt.scatter(data[:, 0], data[:, 1])
    for i, (component1, component2) in enumerate(zip(pca_model.components_[0], pca_model.components_[1])):
        plt.arrow(0, 0, component1, component2, color='r', alpha=0.5)
        if pca_model.components_.shape[0] <= 10:
            plt.text(component1, component2, f"Feature {i+1}", fontsize=12)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Biplot of PCA')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    return filename


# In[55]:


def biplot_dataframe(pca_model, data, labels=None, filename="biplot.png"):
    """
    Create biplot of PCA.

    Parameters:
    - pca_model: PCA model
    - data: Data to be plotted (DataFrame)
    - labels: Labels for data points (optional)
    - filename: Filename to save the plot (default: "biplot.png")

    Returns:
    - Filename of the saved plot
    """
    if labels is not None:
        for label in set(labels):
            indices = labels == label
            plt.scatter(data.loc[indices, data.columns[0]], data.loc[indices, data.columns[1]], label=label)
    else:
        plt.scatter(data[data.columns[0]], data[data.columns[1]])
    for i, (component1, component2) in enumerate(zip(pca_model.components_[0], pca_model.components_[1])):
        plt.arrow(0, 0, component1, component2, color='r', alpha=0.5)
        if pca_model.components_.shape[0] <= 10:
            plt.text(component1, component2, f"Feature {i+1}", fontsize=12)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Biplot of PCA')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    return filename


# In[41]:


def remove_outliers_pca(dataset, pca_model, threshold=2.0):
    """
    Remove outliers from the dataset based on the distance from the centroid in PCA space.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - pca_model: PCA model
    - threshold: Threshold for outlier detection (default: 2)

    Returns:
    - Dataset with outliers removed
    """
    pca_data = pca_model.transform(dataset)
    centroid = np.mean(pca_data, axis=0)
    distances = np.linalg.norm(pca_data - centroid, axis=1)
    # distances = distances.tolist()
    # print("\n\n\n")
    # print("Type: ", type(np.std(distances)))
    # print("Threshold: ", threshold)
    # print("Distances: ", distances)
    # print("Type of distances: ", type(distances))
    # print(np.std(distances))
    # mask = []
    # num=0
    # for i in distances:
    #     num=num+1
    #     if i<float(threshold)*float(np.std(distances)):
    #         mask.append(num-1)
    #     else:
    #         distances.remove(i)
            
    
    # print("MASK: ", str(mask))
    # print("DISTances: ", distances)
    out = threshold * np.std(distances)
    mask = distances < out
    return dataset[mask], mask


# In[56]:


def remove_outliers_pca_dataframe(dataset, pca_model, threshold=2):
    """
    Remove outliers from the dataset based on the distance from the centroid in PCA space.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - pca_model: PCA model
    - threshold: Threshold for outlier detection (default: 2)

    Returns:
    - Dataset with outliers removed (DataFrame)
    - Mask indicating outlier rows
    """
    pca_data = pca_model.transform(dataset)
    centroid = np.mean(pca_data, axis=0)
    distances = np.linalg.norm(pca_data - centroid, axis=1)
    mask = distances < threshold * np.std(distances)
    return dataset[mask], mask


# In[57]:


def get_principal_components(pca_model):
    """
    Retrieve individual principal components from a PCA model.

    Parameters:
    - pca_model: PCA model

    Returns:
    - DataFrame containing the principal components
    """
    principal_components_df = pd.DataFrame(pca_model.components_, columns=pca_model.components_.columns)
    return principal_components_df


# In[42]:


def inverse_transform_component(component, pca_model):
    """
    Perform inverse transformation for an individual principal component.

    Parameters:
    - component: Principal component array
    - pca_model: PCA model

    Returns:
    - Reconstructed feature vector for the component
    """
    return np.dot(component, pca_model.components_)


# In[58]:


def inverse_transform_component_dataframe(component, pca_model):
    """
    Perform inverse transformation for an individual principal component.

    Parameters:
    - component: Principal component array (DataFrame)
    - pca_model: PCA model

    Returns:
    - Reconstructed feature vector for the component (DataFrame)
    """
    return pd.DataFrame(np.dot(component, pca_model.components_), columns=pca_model.components_.columns)


# In[43]:


from sklearn.decomposition import IncrementalPCA

def incremental_pca(dataset, n_components, batch_size=100):
    """
    Apply Incremental PCA to a dataset.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - n_components: Number of components to keep (int)
    - batch_size: Batch size for incremental PCA (default: 100)

    Returns:
    - Transformed dataset after PCA
    - Incremental PCA model
    """
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    transformed_data = inc_pca.fit_transform(dataset)
    return transformed_data, inc_pca


# In[59]:


def incremental_pca_dataframe(dataset, n_components, batch_size=100):
    """
    Apply Incremental PCA to a dataset.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - n_components: Number of components to keep (int)
    - batch_size: Batch size for incremental PCA (default: 100)

    Returns:
    - Transformed dataset after PCA (DataFrame)
    - Incremental PCA model
    """
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    transformed_data = inc_pca.fit_transform(dataset)
    transformed_df = pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(n_components)])
    return transformed_df, inc_pca


# In[44]:


from sklearn.model_selection import cross_val_score

def cross_val_pca(dataset, estimator, cv=5):
    """
    Perform cross-validation for PCA.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - estimator: Estimator to evaluate
    - cv: Number of folds for cross-validation (default: 5)

    Returns:
    - Array of cross-validation scores
    """
    scores = []
    for n in range(1, min(dataset.shape) + 1):
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(dataset)
        score = cross_val_score(estimator, X_pca, cv=cv).mean()
        scores.append(score)
    return np.array(scores)


# In[60]:


def cross_val_pca_dataframe(dataset, estimator, cv=5):
    """
    Perform cross-validation for PCA.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - estimator: Estimator to evaluate
    - cv: Number of folds for cross-validation (default: 5)

    Returns:
    - Array of cross-validation scores
    """
    scores = []
    for n in range(1, min(dataset.shape) + 1):
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(dataset)
        score = cross_val_score(estimator, X_pca, cv=cv).mean()
        scores.append(score)
    return np.array(scores)


# In[45]:


def feature_selection_pca(dataset, target, n_components):
    """
    Perform PCA for feature selection.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - target: Target variable (1D array-like)
    - n_components: Number of components to keep (int)

    Returns:
    - Transformed dataset after PCA
    - Selected features
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(dataset)
    selected_features = pca.components_[:n_components]
    return X_pca, selected_features


# In[61]:


def feature_selection_pca_dataframe(dataset, target, n_components):
    """
    Perform PCA for feature selection.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - target: Target variable (1D array-like)
    - n_components: Number of components to keep (int)

    Returns:
    - Transformed dataset after PCA (DataFrame)
    - Selected features
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(dataset)
    selected_features = pca.components_[:n_components]
    return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)]), pd.DataFrame(selected_features, columns=dataset.columns)


# In[46]:


def cluster_visualization_pca(dataset, labels, pca_model, filename="cluster_visualization_pca.png"):
    """
    Visualize clusters in reduced PCA space and save the plot in the current working directory.

    Parameters:
    - dataset: Input dataset (2D array-like)
    - labels: Cluster labels []-. target col unique
    - pca_model: PCA model
    - filename: Filename to save the plot (default: "cluster_visualization_pca.png")

    Returns:
    - Filename of the saved plot
    """
    pca_data = pca_model.transform(dataset)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization in PCA Space')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    return filename


# In[62]:


def cluster_visualization_pca_dataframe(dataset, labels, pca_model, filename="cluster_visualization_pca.png"):
    """
    Visualize clusters in reduced PCA space and save the plot in the current working directory.

    Parameters:
    - dataset: Input dataset (DataFrame)
    - labels: Cluster labels
    - pca_model: PCA model
    - filename: Filename to save the plot (default: "cluster_visualization_pca.png")

    Returns:
    - Filename of the saved plot
    """
    pca_data = pca_model.transform(dataset)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization in PCA Space')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    return filename


# In[ ]:





# In[ ]:


# Convert Notebook into Script
# get_ipython().system('jupyter nbconvert --to script PCA.ipynb')


# # CHECK

# # Test

# In[63]:


# from sklearn.datasets import fetch_openml


# In[64]:


# # Load the MNIST dataset
# mnist = fetch_openml('mnist_784', version=1, cache=True)

# # Separate features (images) and labels
# X = mnist.data.to_numpy()
# y = mnist.target.astype(int)

# # Display original images
# def plot_images(images, title):
#     fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
#     for ax, image in zip(axes, images):
#         ax.imshow(image.reshape(28, 28), cmap='gray')
#         ax.axis('off')
#     fig.suptitle(title)
#     plt.show()

# # Select random images for display
# random_indices = np.random.choice(len(X), size=10, replace=False)
# random_images = X[random_indices]

# # Plot original random images
# plot_images(random_images, "Original Images")

# # Number of principal components to keep
# n_components = 130

# # Compress and reconstruct data
# compressed_data, reconstructed_data = compress_and_reconstruct(X, n_components)

# # Plot compressed random images
# compressed_images = reconstructed_data[random_indices]
# plot_images(compressed_images, f"Compressed Images (Components={n_components})")

# # Plot explained variance ratio
# pca = PCA(n_components=n_components)
# pca.fit(X)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Cumulative Explained Variance vs Number of Components')
# plt.show()


# In[ ]:




