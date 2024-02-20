""" 
KMeansBalanced algorithm by Malinen, Mikko I., and Pasi Fränti. 
"Balanced k-means for clustering." Structural, Syntactic, and Statistical Pattern Recognition:
Joint IAPR International Workshop, S+ SSPR 2014, Joensuu, Finland, August 20-22, 2014. Proceedings.
Springer Berlin Heidelberg, 2014 and implementation from code at
https://github.com/kayfuku/Implement-K-Means-Clustering-and-Balanced-K-Means
"""
import numpy as np
import pandas as pd
from sklearn.utils.extmath import squared_norm
from munkres import Munkres


class KMeansBalanced:
    def __init__(self, k, max_iterations=100):
        self.n_clusters = k
        self.max_iterations = max_iterations
        self.balanced = True

    def init_centers(self, X):
        shuffled_indices = np.random.permutation(len(X))
        center_indices = shuffled_indices[: self.n_clusters]
        centers = np.zeros(shape=(self.n_clusters, X.shape[1]), dtype=X.dtype)
        for i, idx in enumerate(center_indices):
            centers[i] = X[idx]
        return centers

    def get_labels_and_inertia_extended(self, X, centers):
        cost_matrix = np.zeros(shape=(self.n_samples, self.n_samples), dtype=X.dtype)
        for sample_idx in range(self.n_samples):

            row = []
            for center_idx in range(self.n_clusters):
                dist = 0.0
                dist += np.dot(X[sample_idx], centers[center_idx])
                dist *= -2
                dist += np.dot(X[sample_idx], X[sample_idx])
                dist += np.dot(centers[center_idx], centers[center_idx])
                row.append(dist)
            row = row * self.cluster_size
            row.extend([row[i] for i in range(self.n_samples % self.n_clusters)])
            cost_matrix[sample_idx] = np.array(row)

        m = Munkres()
        indices = m.compute(cost_matrix)

        labels = np.full(self.n_samples, -1, np.int32)
        inertia = 0.0
        for row, column in indices:
            inertia += cost_matrix[row][column]
            labels[row] = column % self.n_clusters

        return labels, inertia

    def move_to_mean(self, X, labels):
        cluster_to_assigned_points = dict()
        for i, cluster in enumerate(labels):
            cluster_to_assigned_points.setdefault(cluster, []).append(X[i])
        cluster_to_mean_point = np.zeros(
            shape=(self.n_clusters, self.n_features), dtype=X.dtype
        )
        for k, v in cluster_to_assigned_points.items():
            cluster_to_mean_point[k] = pd.Series(v).mean()

        return cluster_to_mean_point

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.cluster_size = int(self.n_samples / self.n_clusters)

        centers = self.init_centers(X)

        best_labels, best_inertia, best_centers = None, None, None

        for i in range(self.max_iterations):
            centers_old = centers.copy()
            labels, inertia = self.get_labels_and_inertia_extended(X, centers)
            centers = self.move_to_mean(X, labels)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            center_shift_total = squared_norm(centers_old - centers)
            if center_shift_total == 0:
                break

        if center_shift_total > 0:
            if not self.balanced:
                best_labels, best_inertia = self.get_labels_and_inertia(X, best_centers)
            else:
                best_labels, best_inertia = self.get_labels_and_inertia_extended(
                    X, best_centers
                )

        list_best_centers = []
        for centroid in best_centers:
            list_best_centers.append(list(centroid))

        return list(best_labels), list_best_centers
