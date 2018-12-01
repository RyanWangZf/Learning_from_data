# -*- coding: utf-8 -*-

'''
This is a implementation of Quadratic Discriminant Analysis (QDA) model.

@author: zephyr WANG
'''
import numpy as np
from numpy import linalg
import pdb

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
pi = 3.1415926

def density_old(x,u,sigma):
    '''
    x : m * n matrix
    u : 1 * n vector
    sigma: n * n matrix
    result : 1 * m vector
    this function accept x,u,sigma as parameters and return corresponding probability of N(u,sigma)
    '''
    m,n = x.shape[0], x.shape[1]
    u = np.repeat(u.reshape(1,n),m,axis=0) 
    buff = -0.5*((x-u).dot(linalg.inv(sigma)).dot((x-u).transpose()))
    exp = np.exp(buff)
    C = 1 / np.sqrt(np.power(2*pi,n) * linalg.det(sigma))
    res = np.diag(C*exp)
    return res



def calculate_pro(x,y):
    '''
    calculate the probability of each class
    '''
    n_class = len(np.unique(y))    
    pro = np.zeros(n_class) 
    for i in range(0,n_class):
        pro[i] = len(y[y==(i+1)])/len(y)
    return pro

def calculate_mean(x,y):
    '''
    calculate the mean of each class
    '''
    n_class = len(np.unique(y))
    n_feature = x.shape[1]
    mean_ = np.zeros((n_class,n_feature))
    for i in range(0,n_class):
        mean_i = np.mean(x[(y==(i+1)).flatten()],axis=0)
        mean_[i] = mean_i
    return mean_ 

def calculate_sigma(x,y,u):
    '''
    u: mean of each class [n_class * n_feature]
    calculate the covariance of each class
    '''
    n_class = len(np.unique(y))
    n_feature = x.shape[1]
    sigma = np.zeros((n_class,n_feature,n_feature))
    for i in range(0,n_class):
        ind_i = (y==(i+1)).flatten()
        u_i = np.repeat(u[i].reshape(1,n_feature),x[ind_i].shape[0],axis=0)
        x_i = x[ind_i]
        sigma[i] = 1/x_i.shape[0] * (x_i-u_i).transpose().dot(x_i-u_i)
    return sigma

def density(x,u,sigma):
    '''
    x: data [n_sample * n_feature]
    u: mean of each class [n_class * n_feature]
    sigma: covariance of each class [n_class * n_feature * n_feature]
    '''
    n_class = 11
    n_feature = 10
    m,n = x.shape[0],x.shape[1]
    res = np.zeros((m,n_class))
    for i in range(0,n_class): 
        u_i = u[i].reshape(1,n_feature) # 1,10
        u_i = np.repeat(u_i,m,axis=0) # m,10
        sigma_i = sigma[i] # 10*10
        buff = -0.5*(x-u_i).dot(linalg.inv(sigma_i)).dot((x-u_i).transpose()) # m,m
        exp_ = np.exp(buff)
        C = 1/np.sqrt(np.power(2*pi,n)*linalg.det(sigma_i))
        res[:,i] = np.diag(C*exp_) # m,1
    
    return res # m,n_class

def classify(x_test,pro,u,sigma):
    '''
    after training, use the model to classify x_test, return y_pred
    pro: the probability of each class [n_class,]
    u: the mean of each class [n_class * n_feature]
    sigma: the covariance of each class [n_class * n_feature * n_feature]
    '''
    n_class = 11
    n_feature = 10
    p_x_y = density(x_test,u,sigma) # n_sample,n_class
    pred = p_x_y * pro # n_sample,n_class 
    y_pred = np.argmax(pred,1) # n_sample,1
    return y_pred.reshape(-1,1)

def main():
    
    pro = calculate_pro(x_train,y_train)    
    mean_ = calculate_mean(x_train,y_train) 
    sigma = calculate_sigma(x_train,y_train,mean_)
    y_pred = classify(x_test,pro,mean_,sigma)
    y_pred += 1 # from (0,10) to (1,11), aligned with y_test
    acc = (y_pred == y_test).mean()
    print(acc)
    pdb.set_trace()

    '''
    u = np.zeros((x_train.shape[1],1))
    sigma = np.eye(x_train.shape[1])
    res = density(x_train,u,sigma)
    print(res)
    pdb.set_trace()
    '''
if __name__  == "__main__":
    main()



