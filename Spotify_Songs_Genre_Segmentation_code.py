# importing directory

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

## Load the file into a DataFrame
df=pd.read_csv("spotify dataset.csv")

# data pre-processing 

df.shape

# Check for missing values
df. head()

df.info()

# Check for missing values
df.isnull().sum() 

sns.countplot(x="mode",data=df)
plt.show

sns.histplot(df["playlist_genre"])
plt.xticks(rotation=80)
plt.title("genres")

#pair plot
sns.pairplot(df[['danceability', 'energy', 'loudness', 'acousticness', 'valence']])
plt.show()

#historgam
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='track_popularity', bins=30, kde=True)
plt.title('Track Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

#box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='playlist_genre', y='danceability')
plt.title('Danceability by Genre')
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Danceability')
plt.show()

#correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


#bar plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='playlist_genre')
plt.title('Playlist Genre Counts')
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

#scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='energy', y='valence', hue='playlist_genre')
plt.title('Energy vs. Valence')
plt.xlabel('Energy')
plt.ylabel('Valence')
plt.legend(title='Genre')
plt.show()

plt.scatter(df['danceability'], df['energy'])
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Scatter Plot of Danceability vs. Energy')
plt.show()

# pie charts
df['playlist_genre'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Playlist Genre Distribution')
plt.ylabel('')
plt.show()

#time series plots
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'])
plt.figure(figsize=(12, 6))
plt.plot(df['track_album_release_date'], df['track_popularity'])
plt.xlabel('Release Date')
plt.ylabel('Popularity')
plt.title('Time Series Plot of Popularity Over Time')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Encode categorical variables
label_encoder = LabelEncoder()
df['playlist_genre_encoded'] = label_encoder.fit_transform(df['playlist_genre'])
df['playlist_name_encoded'] = label_encoder.fit_transform(df['playlist_name'])

# Standardize numerical features
scaler = StandardScaler()
num_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
df[num_features] = scaler.fit_transform(df[num_features])

# Select the features for clustering
X = df[['playlist_genre_encoded', 'playlist_name_encoded'] + num_features]


#determine the number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Choosing the optimal number of clusters and perform clustering
n_clusters = 3

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(X)


# Visualize the clusters
# Scatter plot for 'playlist_genre_encoded' vs 'playlist_name_encoded'
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_df = df[df['cluster'] == i]
    plt.scatter(cluster_df['playlist_genre_encoded'], cluster_df['playlist_name_encoded'], label=f'Cluster {i}')
plt.title('Cluster Visualization')
plt.xlabel('Playlist Genre (Encoded)')
plt.ylabel('Playlist Name (Encoded)')
plt.legend()
plt.show()


df['cluster'] = kmeans.labels_


user_cluster = 1 

# Filter playlists from the same cluster as the user's preferences
recommended_playlists = df[df['cluster'] == user_cluster]['playlist_name'].unique()

# Display the recommended playlists
print("Recommended Playlists:")
for playlist in recommended_playlists:
    print(playlist)



