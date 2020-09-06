import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data

def indexing(i):

  labels =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  return labels[i]


class NearestNeighbor:

  def __init__(self):
    pass

  def train(self,X,y):
    self.Xtr=X
    self.ytr=y

  def predict(self,X):
    num_test=X.shape[0]                            # Get figures of array size
    ypred=np.zeros(num_test, dtype=self.ytr.dtype) # In ypred, make array amount of array size with 0 setting type like ytr

    for i in range(20):
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]),axis=1))    # Using L2 distance
      min_index = np.argmin(distances)                                    
      ypred[i]=self.ytr[min_index]
    
    return ypred

(Xtr,ytr),(Xte,yte) = load_data()
Xtr_rows= Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows= Xte.reshape(Xte.shape[0],32*32*3)
nn=NearestNeighbor()
nn.train(Xtr_rows,ytr)
yte_predict=nn.predict(Xte_rows)
print("Accuracy: %f" %np.mean(yte_predict==yte))

rows=4
cols=5
i=1
fig = plt.figure(figsize=(9,9))


for i in range(20):                                       #For visualization
  ax = fig.add_subplot(rows,cols,i+1)
  ax.imshow(Xte[i])
  ax.set_title('%s' %indexing(int(yte_predict[i])))

  ax.set_xticks([])
  ax.set_yticks([])


plt.show()
