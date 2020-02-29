from tensorflow.keras import datasets
from sklearn import neighbors, preprocessing
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from datetime import datetime




# Load the data
data = datasets.mnist.load_data()
# take a sample
n=2000
d=2
# create an array with train data without labels
X_train=data[0][0]
# create an array with the labels of the train data
y_train=data[0][1]
# create a scaler
scaler=preprocessing.MinMaxScaler()
# reshape the train data 6000x784
X_train=X_train.reshape(60000, 28*28)
# use the scaler to scale the data
X_train=scaler.fit_transform(X_train)
# create an array 1x60000 from 0 to 60000
indicators=np.arange(60000)
# suffle the array indicators to randomly choose n samples from the original data
np.random.shuffle(indicators)
# choose the n first indicators
in_samples=indicators[: n]
# choose the data that in_samples indicate
X_samples=X_train[in_samples]
# Choose the corresponding labels
y_samples=y_train[in_samples]


#### IsoMAP ###

# find the k neighbors graph for each instance
t0 = datetime.now()
K_nearest=neighbors.kneighbors_graph(X_samples,300,mode='distance',metric='minkowski', p=2)
# conversion k neighbors graph to numpy array
K_nearest=K_nearest.toarray()
print("K-nearest neighbors matrix:")
print(K_nearest.shape)
# find the shortest path from each datapoint to another
dijkstra=csgraph.dijkstra(K_nearest,directed=False, unweighted=False) 
# calculate the centering matrix H
# where H = I - 1/n*e*e.transpose 
# and e.transpose=[1 1 1 1 ...... 1 1] 1xn
H=np.eye(n)-(1/n)*np.ones((n,n))
# calculate matrix K where K = (-1/2)*H*D^2*H
# where D = dijkstra
K=(-1/2)*(H.dot(dijkstra**2).dot(H))
# eigen decomposition of K
# calculate matrix L with eigenvalues
# calculate matrix V with eigenvectors
L,V=np.linalg.eigh(K)
V=V.T
# take the top 2 eigenvalues with corresponding eigenvectors
top_L=L[-2 :]
top_V=V[-2 :]

for g in range(d):
    top_L[g]=top_L[g]*0.5

# Calculate new the values for represantation of each data point to 2d dimensions
Y=top_L*top_V.T
print(Y)
print(Y.shape)


### Spectral Clustering ###

# find the k neighbors graph for each instance, but this time using matrix Y
# that is, using the new data points in 2d
W=neighbors.kneighbors_graph(Y,300,mode='distance',metric='minkowski', p=2)
# conversion W graph to numpy array
W=W.toarray()

# create a matrix nxn with zeros to use it as a diagonal matrix
Diagonal=np.zeros((n,n))
# We add all elements in every row in matrix W 
# and we save the rult in the diagonal (Diagonal[i][i])
r=range(n)
print(r)
for i in r:
        sum=0
        for j in r:
            sum=sum+W[i][j]
        Diagonal[i][i]=sum

print(Diagonal.shape)
print(Diagonal)

# calculate the Laplacian matrix Laplacian = Diagonal - W
Laplacian=np.subtract(Diagonal, W)
print(Laplacian.shape)
print(Laplacian)

# eigen decomposition of Laplacian
# calculate matrix e_L with eigenvalues
# calculate matrix e_V with eigenvectors
e_L, e_V=np.linalg.eigh(Laplacian)
e_V=e_V.T
# calculate the final array F
# when we want to find k klusters, we need k-1 eigenvectors
F=e_V[1:10]
F=F.transpose()
print(F.shape)

# Kmeans method 
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(F)
print(clusters)
# plot the results
plt.scatter(Y[:, 0], Y[:, 1], c=clusters, s=20, cmap='viridis')
t1 = datetime.now()
print("Time = ",t1-t0)


### accuracy ###

indi_clusters=[]
for i in range(n):
    if clusters[i]==9:
        indi_clusters.append(i)
print(indi_clusters)
value=len(indi_clusters)
print(value)

output=np.zeros(value)

for i in range(len(indi_clusters)):
    value=indi_clusters[i]
    output[i]=y_samples[value]

print(output)
print(output.shape)
print("cluster: 9")
unique_elements, counts_elements = np.unique(output, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))











        
        
        
    
    


    
    
















    




























