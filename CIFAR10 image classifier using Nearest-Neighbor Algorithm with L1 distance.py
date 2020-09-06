#Using CS231n lecture's image classification note,I add some code for visualizaion


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data

def indexing(i):              # Labels are int type from 0~9, so changing the label by using this function.
                              # According this site "https://www.cs.toronto.edu/~kriz/cifar.html", you can use batches.meta file for not doing this. 

  labels =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  return labels[i]


class NearestNeighbor:
  def __init__(self):
    pass

  def train(self,X,y):
    self.Xtr= X
    self.ytr= y

  def predict(self,X):
    num_test=X.shape[0]      
    ypred=np.zeros(num_test, dtype=self.ytr.dtype)

    for i in range(20):
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
      min_index = np.argmin(distances)                          
      ypred[i]=self.ytr[min_index]

    return ypred



(Xtr,ytr),(Xte,yte) = load_data()
Xtr_rows=Xtr.reshape(Xtr.shape[0],32*32*3)                # Xte.shape[0]=10000 pixels=32*32 RGB color=3
Xte_rows=Xte.reshape(Xte.shape[0],32*32*3)                # Xte.shape[0]=10000 pixels=32*32 RGB color=3

nn=NearestNeighbor()                                      #NN Algorithm
nn.train(Xtr_rows,ytr)
yte_predict=nn.predict(Xte_rows)
print ("accuracy: %f" %( np.mean(yte_predict == yte)))

rows=4
cols=5
i=1
fig = plt.figure(figsize=(9,9))


for i in range(20):                                     # for Visualizion.
  ax = fig.add_subplot(rows,cols,i+1)
  ax.imshow(Xte[i])
  ax.set_title('%s' %indexing(int(yte_predict[i])))

  ax.set_xticks([])
  ax.set_yticks([])


plt.show()

