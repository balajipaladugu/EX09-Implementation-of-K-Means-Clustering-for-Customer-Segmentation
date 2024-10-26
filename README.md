# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation. 
2. Choosing the Number of Clusters (K).
3. K-Means Algorithm Implementation.
4. Evaluate Clustering Results.
5. Deploy and Monitor.

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: PALADUGU VENKATA BALAJI
RegisterNumber: 2305001024 


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from  sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],x['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
K=3
Kmeans=KMeans(n_clusters=K)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i], label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points, [centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustring')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/5792393d-d89c-476f-843b-9f3f9991fb8a)
![image](https://github.com/user-attachments/assets/2d5d7f44-33e9-4255-b891-02c5cf11599e)
![image](https://github.com/user-attachments/assets/00992cf5-8482-448a-9944-c0a07fa76c34)
![image](https://github.com/user-attachments/assets/fa797d36-fb18-4886-93ff-33ff175fb896)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
