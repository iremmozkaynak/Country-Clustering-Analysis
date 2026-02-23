from src.preprocess import load_data, scale_data
from src.pca_analysis import apply_pca
from src.clustering import *
from src.visualization import plot_histograms, plot_boxplots, plot_world_map

import pandas as pd

# 1 — Load
df = load_data("data/29-country_data.csv")

# 2 — EDA
plot_histograms(df)

# 3 — Scaling
scaled_df = scale_data(df)

# 4 — PCA
pca_df, pca_model = apply_pca(scaled_df, n_components=3)

# best paramaeters for dbscan
eps, min_sample = find_best_parameters(pca_df)

# 5 — Clustering
print("\n     PCA DATA RESULTS     ")

labels_kmean, score_kmean = run_kmeans(pca_df)
labels_dbscan, score_dbscan = run_dbscan(pca_df,eps=eps, min_samples=min_sample)
labels_hdbscan, score_hdbscan = run_hdbscan(pca_df) 
labels_hierarchical, score_hierarchical = run_hierarchical(pca_df)

print("Silhouette Score (KMeans):", score_kmean)
print("Silhouette Score (DBSCAN):", score_dbscan)
print("Silhouette Score (HDBSCAN):", score_hdbscan)
print("Silhouette Score (Hierarchical):", score_hierarchical)

print("\n     NO PCA DATA RESULTS     ")

labels_kmean2, score_kmean2 = run_kmeans(scaled_df)
labels_dbscan2, score_dbscan2 = run_dbscan(scaled_df ,eps=eps, min_samples=min_sample)
labels_hdbscan2, score_hdbscan2 = run_hdbscan(scaled_df) 
labels_hierarchical2, score_hierarchical2 = run_hierarchical(scaled_df)

print("Silhouette Score (KMeans):", score_kmean2)
print("Silhouette Score (DBSCAN):", score_dbscan2)
print("Silhouette Score (HDBSCAN):", score_hdbscan2)
print("Silhouette Score (Hierarchical):", score_hierarchical2)

# 6 — Comparison Table
results = pd.DataFrame({
    "Method":[
        "KMeans PCA","DBSCAN PCA","HDBSCAN PCA","Hierarchical PCA",
        "KMeans","DBSCAN","HDBSCAN","Hierarchical"
    ],
    "Score":[
        score_kmean,score_dbscan,score_hdbscan,score_hierarchical,
        score_kmean2,score_dbscan2,score_hdbscan2,score_hierarchical2
    ]
})
print("\n     MODEL COMPARISON    ")
print(results)

# 7 — Best Model Selection
best_index = results["Score"].idxmax()
best_method = results.iloc[best_index]["Method"]

print("\nBest Model:", best_method)



# 8 — Assign best labels
best_labels = [
    labels_kmean,labels_dbscan,labels_hdbscan,labels_hierarchical,
    labels_kmean2,labels_dbscan2,labels_hdbscan2,labels_hierarchical2
][best_index]

df["Class"] = best_labels


# 9 — Visualization
plot_boxplots(df)

print("\nShowing maps for all algorithms")

plot_world_map(df, labels_kmean, "KMeans PCA")
plot_world_map(df, labels_hierarchical, "Hierarchical PCA")
plot_world_map(df, labels_dbscan, "DBSCAN PCA")
plot_world_map(df, labels_hdbscan, "HDBSCAN PCA")

