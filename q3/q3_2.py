import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_csv("question-3-features-train.csv")
x_train = df.to_numpy()
df = pd.read_csv("question-3-labels-train.csv")
y_train = df.to_numpy()
df = pd.read_csv("question-3-features-test.csv")
x_test = df.to_numpy()
df = pd.read_csv("question-3-labels-test.csv")
y_test = df.to_numpy()


def normalize(x):
    min_ = np.min(x)
    max_ = np.max(x)
    range_ = max_ - min_

    return [(a - min_) / range_ for a in x]



def initialize_parameters(dimension):
    w = np.random.normal(0.0, 0.01, size=(dimension, 1))
    b = np.random.normal(0.0, 0.01)
    return w,b

def sigmoid(z):
    return 1/((1+np.exp(-z))+0.000001)

def scores(labels, predicted_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for n in range(0,len(labels)):
        if predicted_labels[n] == 1 and labels[n] == 1:
            TP +=1
        if predicted_labels[n] == 0 and labels[n] == 0:
            TN +=1
        if predicted_labels[n] == 1 and labels[n] == 0:
            FP +=1
        if predicted_labels[n] == 0 and labels[n] == 1:
            FN +=1
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    npv =  TN / (TN+FN)
    fpr = FP / (FP +TN)
    fdr = FP / (FP + TP)
    f1 = 2*(recall * precision) / (recall + precision)
    f2 = (5 * precision * recall) / (4 * precision + recall)
    print("accuracy = %.3f " % accuracy)
    print("precision = %.3f " % precision)
    print("recall = %.3f " % recall)
    print("npv = %.3f " % npv)
    print("fpr = %.3f " % fpr)
    print("fdr = %.3f " % fdr)
    print("f1 = %.3f " % f1)
    print("f2 = %.3f " % f2)

    return TP, TN, FP, FN


m = np.shape(x_train)[0]
m_test = np.shape(x_test)[0]
dimension = np.shape(x_train)[1]

x_train = np.array(normalize(x_train))
y_predictions = []
lr = 0.01
def random_mini_batches(X, Y, mini_batch_size = 100, seed = 0):

    
    np.random.seed(seed)           
    m = X.shape[1]                  
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    num_complete_minibatches = math.floor(m / mini_batch_size) 
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k*inc : (k+1)*inc]
        mini_batch_Y = shuffled_Y[:, k*inc : (k+1)*inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, int(m/inc)*inc : ]
        mini_batch_Y = shuffled_Y[:, int(m/inc)*inc : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

mini_batches = random_mini_batches(x_train.T, y_train.T)
w, b = initialize_parameters(dimension)
print("learning rate is " + str(lr))
for i in range(1000):
    for minibatch in mini_batches:
        log_odds = np.array(np.dot(w.T,minibatch[0]) + b , dtype=np.float128)
        y_pred_proba = sigmoid(log_odds)
        log_likelihood = (1/100) * np.dot(minibatch[1],np.log(y_pred_proba.T)) + np.dot((1-minibatch[1]),np.log(1-y_pred_proba.T))
        dw =  np.dot(minibatch[0],(minibatch[1].T - y_pred_proba.T))
        db = np.sum(minibatch[1] - y_pred_proba.T)
        w = w + lr*dw
        b = b + lr*db
    if(i%100==0):
        print("log likelihood in "+ str(i) +" iterations is " + str(log_likelihood[0][0]))
y_predicted_test = np.zeros((m_test, 1))
log_odds_test = np.dot(w.T,x_test.T) + b
y_pred_proba_test = sigmoid(log_odds_test)
for i in range(y_pred_proba_test.shape[1]):
    if y_pred_proba_test[0, i] > 0.5 :
        y_predicted_test[i,0] = 1
    else:
        y_predicted_test[i,0] = 0
y_predictions.append(y_predicted_test)
TP, TN, FP, FN = scores(y_test,y_predicted_test)
print("TP " +str(TP))
print("TN " +str(TN))
print("FP " +str(FP))
print("FN " +str(FN))

mini_batches = random_mini_batches(x_train.T, y_train.T,1)
w, b = initialize_parameters(dimension)
print("learning rate is " + str(lr))
for i in range(1000):
    for minibatch in mini_batches:
        log_odds = np.array(np.dot(w.T,minibatch[0]) + b , dtype=np.float128)
        y_pred_proba = sigmoid(log_odds)
        log_likelihood = np.dot(minibatch[1],np.log(y_pred_proba.T)) + np.dot((1-minibatch[1]),np.log(1-y_pred_proba.T))
        dw =  np.dot(minibatch[0],(minibatch[1].T - y_pred_proba.T))
        db = np.sum(minibatch[1] - y_pred_proba.T)
        w = w + lr*dw
        b = b + lr*db
    if(i%100==0):
        print("log likelihood in "+ str(i) +" iterations is " + str(log_likelihood[0][0]))
y_predicted_test = np.zeros((m_test, 1))
log_odds_test = np.dot(w.T,x_test.T) + b
y_pred_proba_test = sigmoid(log_odds_test)
for i in range(y_pred_proba_test.shape[1]):
    if y_pred_proba_test[0, i] > 0.5 :
        y_predicted_test[i,0] = 1
    else:
        y_predicted_test[i,0] = 0
y_predictions.append(y_predicted_test)
TP, TN, FP, FN = scores(y_test,y_predicted_test)
print("TP " +str(TP))
print("TN " +str(TN))
print("FP " +str(FP))
print("FN " +str(FN))
            

            