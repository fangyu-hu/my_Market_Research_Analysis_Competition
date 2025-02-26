# Assuming 'data' is your DataFrame
columns_1 = data.filter(regex='^6、')

# Data Preprocessing
# Assuming you have only numeric values, otherwise, you might need additional preprocessing
scaler = StandardScaler()
columns_1_scaled = scaler.fit_transform(columns_1)

# Choosing the number of clusters (you may need to adjust this based on your data)
num_clusters = 3

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(columns_1_scaled)

# Adding the cluster labels to the DataFrame
columns_1['Cluster'] = clusters

# Analysis of Cluster Centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columns_1.columns[:-1])

# Displaying the results
# print(columns_1.head())
print(cluster_centers)

# Visualization (you may need to adjust this based on your data)
plt.scatter(columns_1_scaled[:, 0], columns_1_scaled[:, 1], c=clusters, cmap='viridis')
plt.scatter(cluster_centers.iloc[:, 0], cluster_centers.iloc[:, 1], marker='x', s=200, linewidths=3, color='red')
plt.show()


