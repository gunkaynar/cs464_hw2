import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("question-2-features.csv")
df2 = pd.read_csv("question-2-labels.csv")
x_train = df.to_numpy()
y_train = df2.to_numpy()

def add_bias(x):
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1))
    x=np.concatenate((b,x), axis=1)
    return x

def train(train_features, train_values):
    coefficients = np.dot(train_features.T, train_features)
    coefficients = np.linalg.inv(coefficients)
    coefficients =  np.dot(coefficients, train_features.T)
    coefficients =  np.dot(coefficients, train_values)
    return coefficients


def find_predictions(coefficients, features):
  predictions = 0
  x = features.T
  for i in range(coefficients.shape[0]):
      predictions += coefficients[i][0] * x[i]
  return predictions



def plot_curve(features, y_train, predictions):
    plt.plot(features[:,1],y_train,"bo",label='ground truth prices')
    plt.plot(features[:,1],predictions,"ro",label='predicted prices')
    plt.xlabel('lstat', color='#1C2833')
    plt.ylabel('price', color='#1C2833')
    plt.title('lstat vs price curve')
    plt.legend(loc='upper right')
    plt.savefig("plot1.png")
    plt.show()




def find_mse(y_train, predictions):
    sum = 0
    for i in range(len(predictions)):
        dif = y_train[i][0] - predictions[i]
        sum += dif**2
    
    mse = sum / (i+1)
    return mse

lstat = (add_bias(x_train[:,12]))
coefficients = train(lstat,y_train)
print(coefficients)
predictions = find_predictions(coefficients,lstat)
plot_curve(lstat,y_train,predictions)

MSE = find_mse(y_train,predictions)
print(MSE)


lstat2 = np.reshape(np.square(x_train[:,12]),(506,1))
features2 = np.append(lstat,lstat2,axis=1)
coefficients2 = train(features2,y_train)
print(coefficients2)
predictions2 = find_predictions(coefficients2,features2)
plot_curve(features2,y_train,predictions2)
MSE2 = find_mse(y_train,predictions2)
print(MSE2)








