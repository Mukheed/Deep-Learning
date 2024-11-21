#exp-2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
iris=datasets.load_iris()
x=iris.data[:,:2]
y=(iris.target!=0)*1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
perceptron=Perceptron(max_iter=100,eta0=0.1,random_state=42)
perceptron.fit(x_train, y_train)
y_pred=perceptron.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:2f}%")
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,edgecolors='k',marker='o')
plt.xlabel('sepal length(standardized)')
plt.ylabel('sepal width(standardized)')
plt.title('perceptron decision boundary')
x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
z=perceptron.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.3,cmap=plt.cm.Paired)
plt.show()