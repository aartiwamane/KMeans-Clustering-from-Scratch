# K-Means Clustering from Scratch

## Overview

This project demonstrates the implementation of the **K-Means Clustering algorithm from scratch using Python and NumPy**.

K-Means is an Unsupervised Machine Learning algorithm used to group similar data points into clusters based on their features.

This implementation uses the Iris dataset to demonstrate how data points are assigned to clusters based on their distance from cluster centroids.

## Key Features

- Iris dataset loading and preprocessing.
- Feature standardization.
- Random centroid initialization.
- Euclidean distance calculation.
- Nearest centroid-based cluster assignment.
- Centroid updating using mean calculation.
- Iterative model training.
- Convergence detection.
- Within-Cluster Sum of Squares (WCSS) calculation.
- Prediction for new data samples.
- Cluster visualization using Matplotlib.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn (Iris dataset loading only)

**Note:** The K-Means algorithm is implemented manually. Scikit-learn's prebuilt KMeans estimator is not used.

## Algorithm Workflow

1. Load the Iris dataset.
2. Standardize the input features.
3. Initialize K centroids randomly.
4. Calculate Euclidean distances between data points and centroids.
5. Assign each data point to its nearest centroid.
6. Update centroids by calculating the mean of assigned data points.
7. Repeat the assignment and update process until convergence.
8. Calculate inertia to evaluate cluster compactness.
9. Visualize the resulting clusters.

## Mathematical Concepts

### 1. Euclidean Distance

Distance between a data point and centroid:

d(x, c) = √Σ(xᵢ - cᵢ)²

### 2. Centroid Update

The new centroid is calculated as the mean of all points belonging to a cluster:

C = (1/N) Σ Xᵢ

### 3. Inertia

Inertia measures the sum of squared distances between data points and their assigned cluster centroids.

Lower inertia generally indicates more compact clusters for a fixed number of clusters.

## Project Structure

```text
KMeans-Clustering-from-Scratch/
│
├── KMeansClustering.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/aartiwamane/KMeans-Clustering-from-Scratch.git
```

Navigate to the project:

```bash
cd KMeans-Clustering-from-Scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python KMeansClustering.py
```

## Learning Outcomes

- Understanding Unsupervised Machine Learning.
- Implementing clustering algorithms manually.
- Understanding Euclidean distance and centroid calculations.
- Learning iterative optimization and convergence.
- Applying feature standardization.
- Evaluating clustering using inertia.
- Visualizing machine learning results.

## Limitations

- The number of clusters must be specified beforehand.
- Random centroid initialization can influence clustering results.
- K-Means is sensitive to outliers and feature scaling.
- Cluster labels do not directly correspond to actual class labels.

## Future Enhancements

- Implement the Elbow Method to determine the optimal K.
- Implement K-Means++ initialization.
- Compare clustering results with Scikit-learn's KMeans.
- Add Silhouette Score for cluster evaluation.
- Develop an interactive clustering visualization.

## Author

**Aarti Wamane**

M.Tech Computer Engineering

GitHub: https://github.com/aartiwamane

LinkedIn: https://linkedin.com/in/aarti-wamane-5276562a6

---

Developed for educational purposes to understand the internal working of Unsupervised Machine Learning algorithms.

