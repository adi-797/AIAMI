import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

sigma       = np.dot(normalized.T,normalized)/(normalized.shape[0])
U,S,V       = np.linalg.svd(sigma)

Features    = np.arange(7,normalizedTrain.shape[1]+1,2)
clf         = svm.SVC(C = 3.4,gamma = 0.0285,cache_size = 1000)
probMatrix  = []
errMatrix   = []

for iii in Features:
  ZTrain  = np.dot(normalizedTrain , U[:,0:iii])
  ZCV     = np.dot(normalizedCV , U[:,0:iii])
  clf.fit(ZTrain,yTrain)
  err     = []
  for jjj in np.arange(yTrain.size):
    err = np.append(err,clf.predict(ZTrain[jjj,:])[0]==yTrain[jjj])
  errMatrix = np.append(errMatrix,np.mean(err))
  prob = []
  for lll in np.arange(yCV.size):
    prob = np.append(prob,clf.predict(ZCV[lll,:])[0]==yCV[lll])
  probMatrix  = np.append(probMatrix,np.mean(prob))
  print iii

  
plt.xlabel("number of features")
plt.ylabel("Error")
plt.text(60, 0.25, r'C=3.4, $\gamma=0.0285$')
plt.plot(Features,1-probMatrix,Features,1-errMatrix)
plt.show()
