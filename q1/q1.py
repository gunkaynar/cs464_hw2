import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

#read images via pandas
df = pd.read_csv('images.csv')


def plot_faces(pixels):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(48, 48), cmap='gray')
    plt.savefig("allface.png")
    
"""
plot_faces(df)
"""
#dataframe to numpy
pixels = df.to_numpy()


#mean center images
M = np.mean(pixels.T, axis=1)
C = pixels - M
#covariance matrix
V = np.cov(C.T)

#eigenvalues and eigenvectors of covariance matrix 10 components
eigen_Values, eigen_Vectors = np.linalg.eig(V)
indices = np.argsort(eigen_Values)
indices_10 = indices[-10:]
eigen_10 = np.zeros(10)
for j in range(indices_10.shape[0]):
  eigen_10[j] = eigen_Values[indices_10[j]]
  
#PVE of first 10 principal components
total_Variance = sum(eigen_Values)
pve = eigen_10/total_Variance
total_Pve = sum(pve)

#eigenvalues and eigenvectors of covariance matrix n components
def pca_n(V,n):
    eigen_Values, eigen_Vectors = np.linalg.eig(V)
    indices = np.argsort(eigen_Values)
    indices_n = indices[-n:]
    eigen_n = np.zeros(n)
    for j in range(indices_n.shape[0]):
      eigen_n[j] = eigen_Values[indices_n[j]]
      
    #PVE of first 10 principal components
    total_Variance = sum(eigen_Values)
    pve = eigen_n/total_Variance
    total_Pve = sum(pve)
    
    #print out PVE
    print("Total Proportion of variance explained by the first "+ str(n) +" principal components:"+str(total_Pve))

#print out PVE
for j in range(10):
  print("Principal Component "+ str(10-j)+" : "+str(pve[j]) )
print("Total Proportion of variance explained by the first 10 Principal components : "+str(total_Pve))

#PVE for k values
k_values = [1,10,50,100,500]
for k in k_values:
    pca_n(V,k)
    
    
#PCA construction
PC_images = np.zeros((10,48,48))
for j in range(10):
  PC_image = eigen_Vectors[:,indices_10[j]]
  PC_image = np.reshape(PC_image,(48,48))
  PC_images[j,:] = PC_image


#PCA save
for k in range(10):
  B = PC_images[k]
  Image.fromarray((255*B).astype(np.uint8)).save('PC'+str(k+1)+'.png')
  
  
#reconstruction with k values
approx_space = np.zeros((5,2304))
for k in range(len(k_values)):
  PC = np.zeros((k_values[k],2304))
  for j in range(k_values[k]):
    PC[j,:]= eigen_Vectors[:,indices[-j-1]]
  gun = pixels
  mn = (np.mean(gun,axis=0)).reshape(1,2304)
  gun = gun -mn
  first_image = gun[0]
  R = first_image.dot(PC.T)
  approx_space[k,:] = R.dot(PC) + mn 

#save reconstruct
for i in range(5):
  B = approx_space[i].reshape((48,48))
  Image.fromarray((B).astype(np.uint8)).save('PC#'+str(k_values[i])+'.png')
  



  
