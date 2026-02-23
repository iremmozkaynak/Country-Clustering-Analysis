# 🌍 Country Clustering Analysis

Unsupervised machine learning project that groups countries based on socio-economic indicators using multiple clustering algorithms and dimensionality reduction techniques.

---

## 📊 Project Overview
This project explores global country data to identify hidden patterns and similarities between countries.  
Different clustering algorithms are applied and compared both **with PCA** and **without PCA** to evaluate performance and cluster stability.

---

## 🚀 Features
- Data preprocessing & scaling
- PCA dimensionality reduction
- Multiple clustering algorithms:
  - KMeans (auto K selection)
  - Hierarchical Clustering
  - DBSCAN
  - HDBSCAN
- Model comparison using silhouette score
- Automatic best model selection
- Boxplot analysis
- Interactive world map visualization

---

## 🧠 Methodology
1. Load dataset
2. Exploratory Data Analysis
3. Feature scaling
4. PCA transformation
5. Apply clustering algorithms
6. Evaluate using silhouette score
7. Compare results
8. Select best model automatically
9. Visualize clusters

---

## 🤖 Algorithms Compared
| Algorithm | Type | Notes |
|--------|------|------|
KMeans | Centroid-based | Fast and scalable |
Hierarchical | Tree-based | Captures structure |
DBSCAN | Density-based | Detects outliers |
HDBSCAN | Density-based | Adaptive clustering |

---

## 📈 Evaluation Metric
**Silhouette Score**

Measures:
- Cluster cohesion
- Cluster separation

Higher score → better clustering.

---

## 🗂 Project Structure
```
country-clustering-analysis/
│
├── data/
├── src/
│   ├── preprocess.py
│   ├── pca_analysis.py
│   ├── clustering.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone repo:

```
git clone https://github.com/yourusername/country-clustering-analysis.git
cd country-clustering-analysis
```

Install dependencies:

```
pip install -r requirements.txt
```

Run project:

```
python main.py
```

---

## 🌍 Output Example
The project generates:

- Histogram distributions
- Cluster comparison scores
- Boxplot analysis
- Interactive world maps for each algorithm

---

## 🏆 Key Insight
Different clustering algorithms behave differently depending on data structure.  
Density-based methods may classify many countries as noise, indicating that the dataset may not contain clearly density-separated groups.

---

## 📌 Future Improvements
- Hyperparameter optimization
- Cluster interpretation analysis
- Automated reporting
- Dashboard visualization

