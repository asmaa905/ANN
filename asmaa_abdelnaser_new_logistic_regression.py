# -*- coding: utf-8 -*-
"""
Created on Sat May 29 23:20:16 2021

@author: Abo_Elalaa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:18:00 2021

@author: Abo_Elalaa
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

def load_dataset():
    train_dataset = h5py.File('C:\\Users\\Abo_Elalaa\\Downloads\\Compressed\\ANN-2021-main\\ANN-2021-main\\logistic regression\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    #print("*****************",train_set_x_orig)
   # print("*****************",train_set_y_orig)

    test_dataset = h5py.File('C:\\Users\\Abo_Elalaa\\Downloads\\Compressed\\ANN-2021-main\\ANN-2021-main\\logistic regression\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    #print("*****************",test_set_x_orig)
    #print("*****************",test_set_x_orig)
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    print("*****************",classes)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()

#print dataset shapes(dimensions)
print("train set x dim= "+str(train_set_x_orig.shape))
print("train set y dim= "+str(train_set_y_orig.shape))
print("test set x dim= "+str(test_set_x_orig.shape))
print("test set y dim= "+str(test_set_y_orig.shape))

#train data
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_set_y = train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    
#test data
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
test_set_y = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])

#normalize data
train_set_x=train_set_x/255##
test_set_x=test_set_x/255##

def sigmoid(input):
    s = 1/(1 + np.exp(-input))
    
    return s

def initialize(dim):   
    w = np.zeros((dim, 1))#2*1
    b = 0.0
    
    return w, b

# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T , X) + b)                                           # compute activation

    cost = - 1/m * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))       # compute cost
    #grads
    dw= 1/m*np.dot(X,(A-Y).T)  
    db= 1/m*np.sum(A-Y) 
    return dw,db,cost

#optimize 

def optimize(w, b, X, Y, iters, sigma, print_cost = False):
  
    costs = []
    
    for i in range(iters):
        
        dw,db, cost = propagate(w, b, X, Y)
       
        DW = dw
        DB= db
        w = w - sigma * DW
        b = b - sigma * DB
       
        if i % 100 == 0:
            costs.append(cost)
    
    param = {"w": w,
              "b": b}
    
    grad = {"dw": DW,
             "db": DB}
    
    return param, grad, costs

def predict(w, b, X):    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    
    ### make prediction
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###

    for i in range(A.shape[1]):        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] >= 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, iters = 2000, sigma = 0.5, print_cost = False):
    
    #initialize
    w, b = initialize(X_train.shape[0])  

    parameters, grads, costs = optimize(w, b, X_train, Y_train, iters, sigma, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

        ### END CODE HERE ###


    #   accuracy
    print('Neural Network')
    print(" accuracy of test: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print(" accuracy of train: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "sigmoid" : sigma,
         "iters": iters}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, iters = 2000, sigma = 0.005, print_cost = True)


#LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,Normalizer

log_reg = LogisticRegression(C=1000.0, random_state=0)
log_reg.fit(train_set_x.T, train_set_y.T.ravel())

log_reg.coef_.shape

log_reg.coef_
log_reg.intercept_

Y_prediction_test = log_reg.predict(test_set_x.T)
Y_prediction_train = log_reg.predict(train_set_x.T)
Y_prediction_test.shape
#y_pred_prob = log_reg.predict_proba(test_set_x.T)


print('--------------------------------------------')
print("     logistic regression    ")
print('LogisticRegressionModel Train Score is : ' , log_reg.score(train_set_x.T, train_set_y.T))
print('LogisticRegressionModel Train Score is : ' , log_reg.score(test_set_x.T, test_set_y.T))

#AccScore=accurancy_score(Y_prediction_test,Y_prediction_train,normalize=False)

print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print('----------------------------------------------')
# #test on new image
from PIL import Image
import matplotlib.image as plotimg
  
file_name= "C:\\Users\\Abo_Elalaa\\Downloads\\Compressed\\ANN-2021-main\\ANN-2021-main\\logistic regression\\cat_img2.jpg"   # change this to the name of your image file 
   
 # We preprocess the image to fit your algorithm.
 
img = np.array(plotimg.imread(file_name))
test_image = img.reshape(1, img.shape[0]*img.shape[1]*3).T


image_af_predict = predict(d["w"], d["b"], test_image)

plt.imshow(img)
print("y = " + str(np.squeeze(image_af_predict)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(image_af_predict)),].decode("utf-8") +  "\" picture.")
