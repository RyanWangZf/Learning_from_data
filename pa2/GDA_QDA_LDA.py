# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:13:23 2018

@author: XieTianwen
"""

import numpy as np
from numpy import linalg

import pdb

# load vowel data stored in npy
'''
NOTICE:
labels of y are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
'''
x_test = np.load('x_test.npy')
print('x_test\'s shape: {}'.format(x_test.shape))
y_test = np.load('y_test.npy')
print('y_test\'s shape: {}'.format(y_test.shape))
x_train = np.load('x_train.npy')
print('x_train\'s shape: {}'.format(x_train.shape))
y_train = np.load('y_train.npy')
print('y_train\'s shape: {}'.format(y_train.shape))

pi = 3.1415926  # value of pi

'''
x : m * n matrix
u : 1 * n vector
sigma ： n * n mtrix
result : 1 * m vector
the function accept x,u ,sigma as parameters and return corresponding probability of N(u,sigma)
you can choise use it to claculate probability if you understand what this function is doing 
your choice!
'''
def density_old(x,u,sigma):
    n = x.shape[1]
    buff = -0.5*((x-u).dot(linalg.inv(sigma)).dot((x-u).transpose()))
    exp = np.exp(buff)
    C = 1 / np.sqrt(np.power(2*pi,n)*linalg.det(sigma))
    result = np.diag(C*exp)
    return result

def density(x,u,sigma):
    '''
    x: data [n_sample * n_feature]
    u: mean of each class [n_class * n_feature]
    sigma: covariance of each class [n_class * n_feature * n_feature]
    '''
    n_class = u.shape[0]
    n_feature = x.shape[1]
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


'''
class GDA
self.X : training data X
self.Y : training label Y
self.is_linear : True for LDA ,False for QDA ,default True
please make youself konw basic konwledge about python class programming

tips : function you may use
np.histogram(bins = self.n_class)
np.reshape()
np.transpose()
np.dot()
np.argmax()

'''
class GDA():
    def __init__(self, X, Y, is_linear = True):
        self.X = X
        self.Y = Y
        self.is_linear =  is_linear
        self.n_class = len(np.unique(y_train)) # number of class , 11 in this problem 
        self.n_feature = self.X.shape[1]       # feature dimention , 10 in this problem
        
        self.pro = np.zeros(self.n_class)     # variable stores the probability of each class
        self.mean = np.zeros((self.n_class,self.n_feature)) #variable store the mean of each class
        self.sigma = np.zeros((self.n_class,self.n_feature,self.n_feature)) # variable store the covariance of each class
        
        
    def calculate_pro(self):  
        #calculate the probability of each class and store them in  self.pro
        '''
        your code
        
        '''
        #self.pro = 
        y = self.Y
        n_class = self.n_class
        n_class = len(np.unique(y))    
        pro = np.zeros(n_class) 
        for i in range(0,n_class):
            pro[i] = len(y[y==(i+1)])/len(y)
        
        self.pro = pro
                
    def calculate_mean(self):
        #calculate the mean of each class and store them in  self.mean
        '''
        your code
        '''
        #self.mean = 
        y = self.Y
        x = self.X
        n_class = self.n_class
        n_feature = self.n_feature
        mean_ = np.zeros((n_class,n_feature))

        for i in range(0,n_class):
            mean_i = np.mean(x[(y==(i+1)).flatten()],axis=0)
            mean_[i] = mean_i
        
        self.mean = mean_ 
                     
    def calculate_sigma(self):
        #calculate the covariance of each class and store them in  self.sigma
        '''
        your code
        '''

        # self.sigma = 
        x = self.X
        y = self.Y
        u = self.mean
        n_class = self.n_class
        n_feature = self.n_feature
        sigma = np.zeros((n_class,n_feature,n_feature))
        for i in range(0,n_class):
            ind_i = (y==(i+1)).flatten()
            u_i = np.repeat(u[i].reshape(1,n_feature),x[ind_i].shape[0],axis=0)
            x_i = x[ind_i]
            sigma[i] = 1/x_i.shape[0] * (x_i-u_i).transpose().dot(x_i-u_i)
        
        if self.is_linear: # for LDA, sigma for any class is identical
            sigma = np.repeat(np.mean(sigma,axis=0).reshape(-1,n_feature,n_feature),n_class,axis=0)
             
        self.sigma = sigma

    def classify(self,x_test):
        # after training , use the model to classify x_test, return y_pre
        '''
        your code
        '''
        # y_pre = 
        '''
        after training, use the model to classify x_test, return y_pred
        pro: the probability of each class [n_class,]
        u: the mean of each class [n_class * n_feature]
        sigma: the covariance of each class [n_class * n_feature * n_feature]
        '''
        n_class = self.n_class
        n_feature = self.n_feature
        pro = self.pro
        u = self.mean
        sigma = self.sigma
        p_x_y = density(x_test,u,sigma) # n_sample,n_class
        pred = p_x_y * pro # n_sample,n_class 
        y_pred = np.argmax(pred,1) + 1 # n_sample,1
        
        return y_pred.reshape(-1,1)
    
LDA = GDA(x_train,y_train) # generate the LDA model
LDA.calculate_pro()        # calculate parameters
LDA.calculate_mean()
LDA.calculate_sigma()
y_pre = LDA.classify(x_test) # do classify after training
LDA_acc = (y_pre == y_test).mean()
print ('accuracy of LDA is:{:.2f}'.format(LDA_acc))   
    

QDA = GDA(x_train,y_train,is_linear=False) # generate the QDA model
QDA.calculate_pro()                     # calculate parameters
QDA.calculate_mean()
QDA.calculate_sigma()
y_pre = QDA.classify(x_test)          # do classify after training
QDA_acc = (y_pre == y_test).mean()
print ('accuracy of QDA is:{:.2f}'.format(QDA_acc))
