# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pdb

def softmax(y):
    '''
    Input:
        y = np.dot(x,theta): (m,n)*(n,10) = (m,10)
    Output:
        softmax(y): (m,10)
    '''
    exp_y = np.exp(y)
    return exp_y / np.sum(exp_y,axis=1).reshape(-1,1)

def main():
    # set params
    lr = 0.2
    nEpoch = 20
    
    # load data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    x_all,y_all = mnist.train.images,mnist.train.labels
    print("MNIST training images:",mnist.train.images.shape)
    print("MNIST testing  images:",mnist.test.images.shape)
 
    # initialize weights
    theta = np.random.randn(784,10) * 0.005
    
    y_true = np.argmax(y_all,1)
    for i in range(nEpoch):
        x_train,y_train = mnist.train.next_batch(1000)
        h = softmax(np.dot(x_train,theta)) # n_samples,10
        
        grad = (-1/y_train.shape[0]) * np.dot(x_train.T,y_train-h)# 784,10 
        theta = theta - lr * grad # 784,10
        
        pred = np.argmax(softmax(np.dot(x_all,theta)),1) # n_samples,1
        acc = np.float32(pred==y_true).sum()/len(y_true)
        print("{}: {}".format(i,acc))
     
    pdb.set_trace()
        
if __name__ == "__main__":
    
    main()


