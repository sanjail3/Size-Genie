# custom_clustering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.stats import norm


class CustomClustering:
    def __init__(self, df):
        self.df = df.copy()

    def preprocess_data(self, columns_to_drop):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.X = numeric_transformer.fit_transform(self.df.drop(columns_to_drop, axis=1))

    def determine_optimal_clusters(self, max_k=10):
        wcss = []
        silhouette_scores = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.X)
            wcss.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k_elbow = np.argmin(np.diff(np.diff(wcss))) + 2
        optimal_k_silhouette = K[np.argmax(silhouette_scores)]
        optimal_k = (optimal_k_elbow + optimal_k_silhouette) // 2

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(K, wcss, 'bo-')
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Within-cluster sum of squares (WCSS)')
        plt.title('Elbow Method For Optimal K')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, 'bo-')
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method For Optimal K')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return optimal_k

    def apply_clustering(self, optimal_k):
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(self.X)
        return self.df

    def get_probabilistic_value(self, series):
        mu, std = norm.fit(series)
        values = np.linspace(series.min(), series.max(), 1000)
        prob_density = norm.pdf(values, mu, std)
        max_prob_value = values[np.argmax(prob_density)]
        return max_prob_value

    def update_size_chart(self, size_chart):
        clustered_prob_values = self.df.groupby(['Cluster', 'Purchased Size'])['Chest'].apply(
            self.get_probabilistic_value).reset_index(name='Probable Chest Size')
        size_chart_updated = size_chart.copy()

        for i, row in clustered_prob_values.iterrows():
            purchase_size = row['Purchased Size']
            probable_chest_size = row['Probable Chest Size']
            size_chart_updated.loc[size_chart_updated['Brand Size'] == purchase_size, 'Chest'] = probable_chest_size

        return size_chart_updated
