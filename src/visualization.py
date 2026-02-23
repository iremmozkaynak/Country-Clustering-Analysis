import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import numpy as np
import pandas as pd
import plotly.express as px

def plot_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(5*n_cols, 4*n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()


def plot_boxplots(df):
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    sns.boxplot(data=df, x="Class", y="child_mort", ax=ax[0])
    ax[0].set_title("Class vs Child Mortality")

    sns.boxplot(data=df, x="Class", y="income", ax=ax[1])
    ax[1].set_title("Class vs Income")

    plt.show()


def plot_world_map(df, labels, title="Cluster Map"):

    map_df = df.copy()
    map_df["Cluster_ID"] = labels  

    valid_clusters = map_df[map_df["Cluster_ID"] != -1]
    
    if not valid_clusters.empty:
        cluster_means = valid_clusters.groupby("Cluster_ID")["child_mort"].mean()
        sorted_clusters = cluster_means.sort_values().index.tolist()
    else:
        sorted_clusters = []

    label_mapping = {"-1": "Transitional / Outlier"}
    
    descriptions = [
        "High Income (No Aid Needed)", 
        "Developing (Need Support)", 
        "Critical Need"
    ]
    
    for i, cluster_id in enumerate(sorted_clusters):
        if i < len(descriptions):
            label_mapping[str(cluster_id)] = descriptions[i]
        else:
            label_mapping[str(cluster_id)] = f"Sub-category {i+1}"

    map_df["Class_Str"] = map_df["Cluster_ID"].astype(str)
    map_df["Class"] = map_df["Class_Str"].map(label_mapping)
    
    map_df["Class"] = map_df["Class"].fillna(map_df["Class_Str"])

    fig = px.choropleth(
        map_df,
        locationmode="country names",
        locations="country",
        color="Class",
        title=title,
        category_orders={"Class": [
            "Critical Need", 
            "Developing (Need Support)", 
            "High Income (No Aid Needed)", 
            "Transitional / Outlier"
        ]}
    )

    fig.update_geos(fitbounds="locations", visible=True)
    fig.show()