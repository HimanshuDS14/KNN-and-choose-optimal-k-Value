import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report


data = pd.read_csv("classfied.data" )
print(data.head(10))



data.drop(['id'] ,axis=1, inplace=True)
print(data.head(10))



x = np.array(data.drop(["TARGET CLASS"] , axis=1))
y = np.array(data["TARGET CLASS"])

print(x)
print(y)


train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.3 , random_state=0)
classifier = KNeighborsClassifier()

classifier.fit(train_x , train_y)


z = np.array([[0.9139173265804122,1.162072707738686,0.5679458536608835,0.7554638959888053,0.7808615715474211,0.3526077229335367,0.7596969140337959,0.6437975644205896,0.8794220913503251,1.2314094373345865] , [1.2342044015229892,1.3867262910227907,0.6530463056350606,0.8256244452701601,1.142503540047211,0.8751279212294097,1.4097080602602086,1.3800025459668137,1.5226920462447513,1.1530930248076359]])

print(classifier.predict(z))


print('***************************************')
error_rate = []

for i in range(1,40):

    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(train_x , train_y)
    pred_i = classifier.predict(test_x)
    error_rate.append(np.mean(pred_i != test_y))




x = [i for i in range(1,40)]
plt.scatter(x,error_rate , color = "red" )
plt.plot(x , error_rate , color = "blue" , linestyle = "dashed")
plt.title("Error rate")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x , train_y)
pred_y = knn.predict(test_x)
print(confusion_matrix(test_y , pred_y))
print(classification_report(test_y , pred_y))


#choose optimal k value is 27
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(train_x , train_y)
pred_y = knn.predict(test_x)
print(confusion_matrix(test_y , pred_y))
print(classification_report(test_y , pred_y))






















