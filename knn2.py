import pandas as pd
import numpy as np
from sklearn import model_selection , neighbors
from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
print(data.head(10))

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values
print(x)
print(y)

train_x , test_x , train_y , test_y = model_selection.train_test_split(x,y,test_size=0.3 , random_state=0)

print(train_x)
print(test_x)

train_array = np.array(train_x)
test_array = test_x
print("**********************************")


classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_x , train_y)
pred = classifier.predict(test_x)


print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))




error_rate = []

for i in range(1,40):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x , train_y)
    pred = knn.predict(test_x)
    error_rate.append(np.mean(pred != test_y))

x = [i for i in range(1,40)]
plt.scatter(x , error_rate , color = "red")
plt.plot(x,error_rate , color = "green" , linestyle = "dashed")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()


#choose optimal k value

knn = neighbors.KNeighborsClassifier(n_neighbors=17)
knn.fit(train_x , train_y)
pred = knn.predict(test_x)
print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))













