from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def elbow_method(data, k_range=10):
    wcss = []
    for k in range(1, k_range + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(data)
        wcss.append(model.inertia_)
    return wcss

def run_kmeans(data, k=3):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score

def run_dbscan(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    
    real_clusters = set(labels) - {-1}
    
    if len(real_clusters) > 1:
        mask = labels != -1
        if len(set(labels[mask])) > 1: 
            score = silhouette_score(data[mask], labels[mask])
        else:
            score = -1
    else:
        score = -1
        
    return labels, score

def run_hdbscan(data, min_cluster_size=5):
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(data)
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1
    return labels, score

def run_hierarchical(data, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score

def find_best_parameters(data):
    eps_values = np.arange(0.1, 1.0, 0.05) 
    min_samples_values = [2, 3, 4] 
    
    best_score = -1
    best_eps = None
    best_min_samples = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            labels, score = run_dbscan(data, eps, min_samples)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

    if best_eps is None:
        print("Uyarı: DBSCAN en az 2 gerçek küme bulamadı. Parametreleri genişletin.")
        return 0.5, 3 
        
    return best_eps, best_min_samples