python implementation of binary classification from scratch. i used numpy as only dependency.in this repo we are going to use the Pima Indians onset of diabetes dataset. you can download dataset from [data.csv](data.csv). It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.
## 1) [dni.py](dni.py)
in this file I'm using decoupled neural interfaces using synthetic gradients  to train the classifier.note that in this classifier i am not using any regularization technique. you can use dropout,  batch_norm or(L1, L2) regularization technique.  but you can use any of those technique,  for dropout you can use [nn.py](nn.py) as reference
## 2) [nn.py](nn.py) 
in this file i'm  using fully connected layer for binary classification, implemented dropout as regularization technique and sigmoid as activation function. you can also use relu in hidden layer for relu activation use [dni.py](dni.py) as refrence 
## 3) [data.csv](data.csv)
this csv file is data set for our binary classification.last column in csv file is labels of the inputs
