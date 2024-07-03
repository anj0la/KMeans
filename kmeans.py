"""

File: kmeans.py

Author: Anjola Aina
Date: March 13th, 2024

Description: 

This file implements the K-means algorithm for image compression. Given a 24-bit color representation of an image, where each pixel is represented as 
three 8-bit unsigned integers and the range is between 0 and 255 (Red, Green, Blue), the file reduces the thousands of colours in the image to 16 colors.
The resulting image is the compressed image of the original image.

Functions:

    - preprocess_data(input_img) -> This function accepts the input image and pre-process it by transforming the three-dimensional matrix into a two-dimensional one.
    - kMeans_init_centroid(k_clusters, X) -> This function takes the input image and number of centroids K and returns the list of centroids.
    - run_kMeans(X, centroids, max_iter) -> This function takes the pre-processed image, centroids and max_iter and returns the centroids, idx (i.e., the corresponding index).
    - find_closest_centroids(X, centroids) -> This function computes the centroid memberships for every example, by taking the input matrix X and centroids. 
    - compute_centroids(X, idx, k) -> This function returns the new centroids by computing the means of the data points assigned to each centroid. It takes the input matrix, the index array and the number of centroids K. 
    - compress_image(X, centroids, idx) -> This function assigns each pixel to its closest centroid. It takes the input matrix, the centroids, and the index array.

Running the File:

To run the file, change the filename where the input image is stored, and where the final image should be saved.
"""
import numpy as np
from PIL import Image as img

def preprocess_data(input_img):
    """
    This function accepts the input image and pre-process it by transforming the three-dimensional matrix into a two-dimensional one.
    It sets the range of the pixels between 0 and 1 by dividing the input matrix by 255, and reshapes the input matrix to flatten the dimension.
    The resulting matrix is of the shape (m, n), instead of (m1, m2, n), where m = m1 x m2.
    
    Args:
        input_img (ndarray): the input image, represented as a three-dimensional array

    Returns:
        ndarray: the resulting flattened input matrix, represented as a two-dimensional matrix
    """
    scaled_image = input_img / 255
    return scaled_image.reshape(-1, 3)

def k_means_init_centroid(X, k_clusters):
    """
    This function takes the input image and number of centroids K and returns the list of centroids.

    Args:
        X (ndarray): the processed input image
        k_clusters (int): the number of centroids (k = 16)

    Returns:
        ndarray: the K centroids
    """
    return X[np.random.choice(X.shape[0], k_clusters, replace=False)]
    
def run_k_means(X, centroids, max_iter):
    """
    This function takes the pre-processed image, centroids and max_iter and returns the centroids, idx (i.e., the corresponding index).
    It uses two helper methods, find_closest_centroids, and compute_centroids, and stops if the maximum number of iterations has been reached,
    or convergence has been reacged (i.e., prev_centroids = centroids).
    
    Args:
        X (ndarray): the processed input image
        centroids (ndarray): the centroids K
        max_iter (int): the max amount of iterations

    Returns:
        (ndarray, ndarray): a tuple containing the centroids and the idx (i.e., the corresponding index) array     
    """
    # initially filling prev_centroids to a np array of zeros with same shape as centroids
    prev_centroids = np.zeros(centroids.shape) 
    curr_centroids = centroids
    
    for _ in range(max_iter):
        # convergence checker, stop if prev_centroids = centroids
        if np.array_equal(prev_centroids, curr_centroids):
                print('WE MADE IT, HA!')
                break
        # get the idx of the closest centroids
        idx = find_closest_centroids(X, curr_centroids)
        # setting prev_centroid to the current centroids before updating it
        prev_centroids = curr_centroids
        curr_centroids = compute_centroids(X, idx, len(centroids))
    # at this point, we have found the top 16 colours and can return the centroids along with the corresponding index array     
    return curr_centroids, idx
               
def find_closest_centroids(X, centroids):
    """
    This function computes the centroid memberships for every example, by taking the input matrix X and centroids. 
    
    Args:
        X (ndarray): the processed input image
        centroids (ndarray): the centroids K

    Returns:
        ndarray: one-dimension array idx which has the same number of elements as X that holds the index of the closest centroid
    """
    idx = np.zeros(len(X))
    # iterating through every data point (training example) in the training set
    for i in range(len(X)):
        closest_distance = np.inf # initialize closest distance to inf for every training example
        for j in range(len(centroids)):
            distance = euclidean_distance(X[i], centroids[j])
            # update the closest centroid if the distance is smaller
            if distance < closest_distance: 
                closest_distance = distance
                idx[i] = j # index of closest centroid is assigned to the current example
    return idx
    
def compute_centroids(X, idx, k_clusters):
    """
    This function returns the new centroids by computing the means of the data points assigned to each centroid. It takes the input matrix, the index array and the number of centroids K. 

    Args:
        X (ndarray): the processed input image
        idx (ndarray): one-dimension array idx holding the index of the closest centroid
        k_clusters (int): the number of centroids (k = 16)

    Returns:
        ndarray: the new centroids
    """
    computed_centroids = np.zeros((k_clusters, X.shape[1])) # X.shape[1] = n, which is 3
    k_clusters = [[] for _ in range(k_clusters)]
     
    # assigning the data points to a specific cluster (each assigned to the centroid closest to it)
    for i in range(len(X)):
        nearest_centroid_idx = int(idx[i])
        k_clusters[nearest_centroid_idx].append(X[i]) 
    # calculating the means for each cluster and updating the centroids 
    for i in range(len(k_clusters)):
        k_cluster_mean = np.mean(k_clusters[i], axis=0)
        computed_centroids[i] = k_cluster_mean
    return computed_centroids
             
def compress_image(X, centroids, idx):
    """
    This function assigns each pixel to its closest centroid. It takes the input matrix, the centroids, and the index array.

    Args:
        X (ndarray): the processed input image
        centroids (ndarray): the centroids K
        idx (ndarray): one-dimension array idx holding the index of the closest centroid

    Returns:
        ndarray: the compressed image 
    """
    for i in range(len(X)):
        final_centroid_idx = int(idx[i])
        X[i] = centroids[final_centroid_idx]
    return np.round(X * 255).astype(np.uint8)

def euclidean_distance(a, b):
    """
    Implements the eucilean distance used to calcuate the distance between 
    two values.

    Args:
        a (Any): first value
        b (Any): second value

    Returns:
        Any: eucilean distance between value a and value b
    """
    return np.sqrt(np.sum(np.square(a - b)))
            
# Running the KMeans algorithm
input_image = np.asarray(img.open('bird_image.png'))

processed_image = preprocess_data(input_image)
centroids = k_means_init_centroid(processed_image, 16)
idx = find_closest_centroids(processed_image, centroids)
final_centroids, final_idx = run_k_means(processed_image, centroids, 100)

compressed_image = compress_image(processed_image, final_centroids, final_idx)
reshaped_compressed_image = compressed_image.reshape(input_image.shape)

final_image = img.fromarray(reshaped_compressed_image)
final_image.save('bird_image_final.png')

print('Finished execution.')