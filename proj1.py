import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv('customer_purchases.csv')

# Data Cleaning
data.dropna(inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(10, 5))
sns.lineplot(data=data, x='date', y='sales')
plt.title('Sales Over Time')
plt.show()

# Clustering
features = data[['customer_age', 'total_spent']]
kmeans = KMeans(n_clusters=3)
data['segment'] = kmeans.fit_predict(features)

# Visualize Segments
sns.scatterplot(data=data, x='customer_age', y='total_spent', hue='segment')
plt.title('Customer Segmentation')
plt.show()

