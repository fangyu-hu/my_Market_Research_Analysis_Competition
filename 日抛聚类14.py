from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

# Assuming 'data' is your DataFrame with columns_3
columns_3 = data.filter(regex='^14、')

# Standardize the data
scaler = StandardScaler()
columns_3_scaled = scaler.fit_transform(columns_3)

# Choose the number of clusters (you may adjust this)
n_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(columns_3_scaled)

# Add cluster labels to the DataFrame
columns_3_clustered = pd.concat([columns_3, pd.Series(clusters, name='Cluster')], axis=1)

# Display clustered data
print(columns_3_clustered.head())

# Feature importances based on cluster means
cluster_means = columns_3_clustered.groupby('Cluster').mean()
feature_importance = cluster_means.T
feature_importance['Total'] = feature_importance.sum(axis=1)
feature_importance = feature_importance.sort_values(by='Total', ascending=False).drop('Total', axis=1)

# Display feature importances
print("Feature Importances:")
print(feature_importance)