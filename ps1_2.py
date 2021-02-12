from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import sklearn
 

#necessary for us to use sklearn.
if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
 
else:
        from sklearn.model_selection import train_test_split
 

mnist = datasets.load_digits()
 
#divides the data set 75% training and %25 testing.
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
        mnist.target, test_size=0.25, random_state=42)
 
#it takes 10% of the testing to verify the output values.
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)
 


print("training : {}".format(len(trainLabels)))
print("validation: {}".format(len(valLabels)))
print("testing: {}".format(len(testLabels)))


kVals = range(1, 50, 2)
accuracies = []
 

for k in range(1, 50, 2):
      
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)
 

