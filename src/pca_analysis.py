import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df, n_components=3):
    pca = PCA()
    transformed = pca.fit_transform(df)
    pca_df = pd.DataFrame(transformed)

    pca_df = pca_df.iloc[:, :n_components]

    return pca_df, pca
