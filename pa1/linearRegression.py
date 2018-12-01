# -*- coding: utf-8 -*-

import numpy as np
import pdb

def main():
    '''
    the linear observation model is y = Ax + b
    learn the x that minimize the loss function 1/2||(Ax-y)||_2^2
    '''
    # set params
    lr = 0.0001
    nEpoch = 1000

    m = 10000 # num of samples
    n = 10    # num of features
    
    # data initialize
    xd = np.random.rand(n,1)
    A = 10*np.random.randn(m,n)
    b = np.random.randn(m,1)
    y = np.dot(A,xd) + b
    
    xl = np.random.randn(n,1)

    for i in range(nEpoch):
        
        #h = np.dot(A,xl)
        #lost = y - h
        #grad = -2/m * np.dot(A.T,lost)
        grad = 2/m * np.dot(A.T,np.dot(A,xl)-y) # notice whether the grad is negative or positive
        xl = xl - lr * grad
        xLoss = np.sum(np.square(xl - xd))
        print("epch {}, loss {}".format(i,xLoss))
   
    print("x data \n",xd)
    print("x learn \n",xl)

    pdb.set_trace()


if __name__ == "__main__":
    
    main()



    

