import numpy as np
from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data') # read data
df.replace('?', -99999, inplace=True) # replace missing data with outlier')
df.drop(['id'], 1, inplace=True) # drop id column'
X = np.array(df.drop(['class'], 1)) # features
y = np.array(df['class']) # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data into training and testing samples

clf =neighbors.KNeighborsClassifier() # classifier
clf.fit(X_train, y_train) # train classifier

accuracy = clf.score(X_test, y_test) # test classifier
print(accuracy)