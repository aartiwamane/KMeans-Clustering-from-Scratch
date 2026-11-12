
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def Marvellous_Header(step, title):
    print("\n" + "=" * 70)
    print(f"Step {step}: {title}")
    print("=" * 70)


# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

Marvellous_Header(1, "Loading Iris Dataset")

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])
print("Target Classes:", iris.target_names)


# ------------------------------------------------------------
# Step 2: Feature Standardization
# ------------------------------------------------------------

Marvellous_Header(2, "Feature Standardization")

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std

print("Original Dataset Mean:")
print(mean)

print("\nStandardized Dataset Mean:")
print(np.mean(X_scaled, axis=0))


# ------------------------------------------------------------
# Step 3: Initialize K-Means Model
# ------------------------------------------------------------

Marvellous_Header(3, "K-Means Model Initialization")


class KMeans:

    def __init__(self, k=3, max_iterations=100, tolerance=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia = None

    # --------------------------------------------------------
    # Calculate Euclidean Distance
    # --------------------------------------------------------

    def calculate_distance(self, point, centroid):

        return np.sqrt(np.sum((point - centroid) ** 2))

    # --------------------------------------------------------
    # Assign Data Points to Nearest Centroid
    # --------------------------------------------------------

    def assign_clusters(self, X):

        labels = []

        for point in X:

            distances = []

            for centroid in self.centroids:

                distance = self.calculate_distance(
                    point, centroid
                )

                distances.append(distance)

            labels.append(np.argmin(distances))

        return np.array(labels)

    # --------------------------------------------------------
    # Update Centroids
    # --------------------------------------------------------

    def update_centroids(self, X, labels):

        new_centroids = []

        for cluster in range(self.k):

            cluster_points = X[labels == cluster]

            if len(cluster_points) > 0:

                centroid = np.mean(cluster_points, axis=0)

            else:

                centroid = self.centroids[cluster]

            new_centroids.append(centroid)

        return np.array(new_centroids)

    # --------------------------------------------------------
    # Calculate Within-Cluster Sum of Squares
    # --------------------------------------------------------

    def calculate_inertia(self, X, labels):

        total_distance = 0

        for cluster in range(self.k):

            cluster_points = X[labels == cluster]

            if len(cluster_points) > 0:

                distances = cluster_points - self.centroids[cluster]

                total_distance += np.sum(distances ** 2)

        return total_distance

    # --------------------------------------------------------
    # Train K-Means Model
    # --------------------------------------------------------

    def fit(self, X):

        np.random.seed(42)

        random_indices = np.random.choice(
            len(X), self.k, replace=False
        )

        self.centroids = X[random_indices]

        print("Initial Centroids:")
        print(self.centroids)

        for iteration in range(self.max_iterations):

            old_centroids = self.centroids.copy()

            # Assign clusters
            self.labels = self.assign_clusters(X)

            # Update centroids
            self.centroids = self.update_centroids(
                X, self.labels
            )

            # Calculate centroid movement
            movement = np.linalg.norm(
                self.centroids - old_centroids
            )

            print(
                f"Iteration {iteration + 1}: "
                f"Centroid Movement = {movement:.6f}"
            )

            # Check convergence
            if movement < self.tolerance:

                print("\nModel Converged Successfully!")
                break

        # Final cluster assignment
        self.labels = self.assign_clusters(X)

        self.inertia = self.calculate_inertia(
            X, self.labels
        )

        return self

    # --------------------------------------------------------
    # Predict Cluster for New Data
    # --------------------------------------------------------

    def predict(self, X):

        return self.assign_clusters(X)


# ------------------------------------------------------------
# Step 4: Train K-Means Model
# ------------------------------------------------------------

Marvellous_Header(4, "Training K-Means Clustering Model")

model = KMeans(
    k=3,
    max_iterations=100,
    tolerance=0.0001
)

model.fit(X_scaled)


# ------------------------------------------------------------
# Step 5: Display Cluster Information
# ------------------------------------------------------------

Marvellous_Header(5, "Cluster Information")

print("Final Centroids:")
print(model.centroids)

print("\nCluster Distribution:")

for cluster in range(model.k):

    count = np.sum(model.labels == cluster)

    print(f"Cluster {cluster}: {count} Samples")


# ------------------------------------------------------------
# Step 6: Calculate Inertia
# ------------------------------------------------------------

Marvellous_Header(6, "Model Evaluation")

print(f"Within-Cluster Sum of Squares: {model.inertia:.4f}")

print(
    "Note: Lower inertia indicates tighter clusters, "
    "but it generally decreases as K increases."
)


# ------------------------------------------------------------
# Step 7: Predict New Data
# ------------------------------------------------------------

Marvellous_Header(7, "Predicting Cluster for New Samples")

new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
])

new_samples_scaled = (new_samples - mean) / std

predictions = model.predict(new_samples_scaled)

for index, prediction in enumerate(predictions):

    print(
        f"Sample {index + 1}: "
        f"Assigned to Cluster {prediction}"
    )


# ------------------------------------------------------------
# Step 8: Cluster Visualization
# ------------------------------------------------------------

Marvellous_Header(8, "K-Means Cluster Visualization")

plt.figure(figsize=(9, 6))

for cluster in range(model.k):

    cluster_points = X_scaled[
        model.labels == cluster
    ]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster}"
    )

plt.scatter(
    model.centroids[:, 0],
    model.centroids[:, 1],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")

plt.title("K-Means Clustering on Iris Dataset")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# Step 9: Compare Cluster Distribution with Actual Classes
# ------------------------------------------------------------

Marvellous_Header(9, "Actual Class and Cluster Comparison")

for cluster in range(model.k):

    actual_classes = y[model.labels == cluster]

    print(f"\nCluster {cluster}:")

    for class_id, class_name in enumerate(iris.target_names):

        count = np.sum(actual_classes == class_id)

        print(f"{class_name}: {count}")
