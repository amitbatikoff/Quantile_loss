import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import umap.umap_ as umap  # Changed UMAP import

def analyze_data(X, y, file_prefix=""):
    """
    Analyzes input and target data, performs dimensionality reduction, clustering,
    and generates visualizations to identify potential issues.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.
        file_prefix (str): Prefix for saving the plots.
    """

    # Make a copy to avoid modifying the original data
    X = X.copy()
    
    # Additional check: plot missing values heatmap if there are missing entries
    if X.isnull().values.any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(X.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title('Missing Values Heatmap')
        plt.savefig(f'{file_prefix}missing_values_heatmap.png')
        plt.close()

    # 1. Visualize Input and Target Data
    plt.figure(figsize=(10, 6))
    plt.scatter(X.index, y, alpha=0.5)
    plt.title('Target Variable vs. Index')
    plt.xlabel('Index')
    plt.ylabel('Target')
    plt.savefig(f'{file_prefix}target_vs_index.png')
    plt.close()

    # 2. Reduce Dimensionality with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca_df['PC1'], X_pca_df['PC2'], alpha=0.5)
    plt.title('PCA: 2 Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{file_prefix}pca_scatter.png')
    plt.close()

    # --- New UMAP Visualization ---
    reducer = umap.UMAP(n_components=2, random_state=42)  # Updated UMAP usage
    X_umap = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5)
    plt.title('UMAP: 2 Components')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(f'{file_prefix}umap_scatter.png')
    plt.close()
    # --- End of UMAP Visualization ---

    # 3. Cluster Data with KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init
    X['Cluster'] = kmeans.fit_predict(X)

    # Visualize clusters in PCA space
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.drop('Cluster', axis=1))
    X_pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    X_pca_df['Cluster'] = X['Cluster']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=X_pca_df, palette='viridis')
    plt.title('Clusters in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{file_prefix}clusters_pca.png')
    plt.close()

    X.drop('Cluster', axis=1, inplace=True)

    # 4. Additional Data Analysis
    # Distribution of Target Values
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True)
    plt.title('Distribution of Target Values')
    plt.xlabel('Target Value')
    plt.savefig(f'{file_prefix}target_distribution.png')
    plt.close()

    # Scatter plots of feature pairs
    num_features = X.shape[1]
    if num_features > 1:
        n_cols = min(3, num_features)
        n_rows = (num_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        for i in range(num_features):
            ax = axes[i]
            sns.scatterplot(x=X.iloc[:, i], y=y, ax=ax)
            ax.set_xlabel(X.columns[i])
            ax.set_ylabel('Target')
            ax.set_title(f'{X.columns[i]} vs. Target')

        # Remove any unused subplots
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(f'{file_prefix}feature_vs_target_scatter.png')
        plt.close()
    else:
        print("Only one feature, skipping feature vs target scatter plots.")

    # 5. Additional Visualizations for Data Diagnostics
    # Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(f'{file_prefix}correlation_matrix.png')
    plt.close()

    # Boxplot for each feature to spot outliers
    plt.figure(figsize=(10, 6))
    X.boxplot()
    plt.title('Features Boxplot')
    plt.savefig(f'{file_prefix}features_boxplot.png')
    plt.close()

    print("Data analysis complete.  Check the generated plots.")

if __name__ == '__main__':
    # Generate some dummy data for demonstration
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.rand(100))

    analyze_data(X, y, "dummy_")
