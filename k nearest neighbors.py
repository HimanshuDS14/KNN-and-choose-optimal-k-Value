import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing  , model_selection , neighbors
from sklearn.metrics import confusion_matrix,classification_report

data = pd.read_csv("breast-cancer-wisconsin.data")
print(data.head(10))

#handle missing data

data.replace('?' , -99999 , inplace=True)

#drop useless data
data.drop(['id'] , axis=1 , inplace=True)

print(data.head(10))

x = np.array(data.drop(["Class"] , axis=1)) #feature is everything except class column
y = np.array(data["Class"])    #label is class column


train_x , test_x , train_y , test_y = model_selection.train_test_split(x,y , test_size=0.2  ,random_state=0)

classifier=  neighbors.KNeighborsClassifier()
classifier.fit(train_x , train_y)


accuracy = classifier.score(test_x , test_y)
print(accuracy)

example_measure = np.array([4,8,7,10,4,10,7,5,1])
example_measure = example_measure.reshape(1,-1)

prediction = classifier.predict(example_measure)
print(prediction)

error_rate = []

for i in range(1,40):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=i)
    classifier.fit(train_x , train_y)
    pred_i = classifier.predict(test_x)
    error_rate.append(np.mean(pred_i != test_y))

x = [i for i in range(1,40)]
plt.scatter(x , error_rate , color = "red")
plt.plot(x,error_rate ,color = "green")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x , train_y)
pred = knn.predict(test_x)

print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))

#choose optimal k value
knn = neighbors.KNeighborsClassifier(n_neighbors=15)
knn.fit(train_x , train_y)
pred = knn.predict(test_x)

print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))

