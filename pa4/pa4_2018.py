# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:15:03 2018

@author: XieTianwen
@assignment_author: WANG Zifeng [On Sat. Dec. 01 09:46 8102]
"""

import numpy as np
import matplotlib.pyplot as plt

def power_iteration(M,eta=1e-7,max_iter_N=100):
    """
    M: the matrix for eigen vector decomposition [np.array]
    eta: minimum tolerance
    max_iter_N: max iteration times for computing
    use power iteration to get the largest eigen vector of matrix M
    """
    # initialize x
    x = np.random.randn(M.shape[0])
    lam = 0
    
    for k in range(max_iter_N):
        y = x / x.max()
        x = M.dot(y)
        beta = x.max()
        err = abs(beta - lam)
        print("{}/{} err:{}".format(k+1,max_iter_N,err))
        y = x / beta
        if err <= eta:
            print("Power Iteration Stop! [Success]")
            break
        else:
            lam = beta
        if k >= (max_iter_N - 1):
            print("Exceed Max Iteration! [Failure]")
    return beta,y # eigen value, normalized eigen vector

def orthogonal_vec(a):
    """
    get an orthogonol vector of vector a
    """
    b = np.ones_like(a)
    a_m = a.T.dot(a)
    oc = b - a.T.dot(a.T.dot(b)/a_m)
    return oc

def second_sv_fn(B,s_py,max_iter_N=10):
    """
    B: matrix
    s_py: vector of sqrt_P of y
    get the 2nd largest sigular vector of B
    """
    norm = lambda x:x/np.sqrt(x.dot(x.T)) # normalizer
     
    psi_1 = orthogonal_vec(s_py)

    for i in range(max_iter_N):
        psi_0 = psi_1
        
        phi_1 = B.dot(psi_1)
        phi_1 = phi_1 / phi_1.max()

        psi_1 = phi_1.T.dot(B)
        phi_1 = phi_1 / phi_1.max()
        
        err = np.mean((abs(psi_0-psi_1)))
        print("{}/{} err psi {}:".format(i+1,max_iter_N,err))

    phi_1 = norm(phi_1)
    psi_1 = norm(psi_1)
    # U,_,V = svd(B)
    # psi_1 == V[1,;]
    # phi_1 == U[:,1]

    return phi_1,psi_1

def main():

    plt.style.use('ggplot')

    num = np.load('num.npy')

    '''
    your codes
    '''

    # acquire matrix B
    Pxy = num / np.sum(num)
    Px  = np.sum(num,1) / np.sum(num)
    Py  = np.sum(num,0) / np.sum(num)
    sqrt_Px = np.sqrt(Px)
    sqrt_Py = np.sqrt(Py)

    sqrt_Px_prime = np.sqrt(np.repeat(Px,15).reshape(10,15))
    sqrt_Py_prime = np.sqrt(np.repeat(Py,10).reshape(15,10).T)
    sqrt_Px_Py = sqrt_Px_prime * sqrt_Py_prime
    matrix_B = Pxy / sqrt_Px_Py # B = Pxy(X,Y) / sqrt(Px(X)*Py(Y))

    # acquire eigen vector of BB^T:Psi(y) and B^TB:Phi(x) with maximum eigen value by Power Iteration
    _,psi = power_iteration(matrix_B.dot(matrix_B.T)) # Psi_1
    _,phi = power_iteration(matrix_B.T.dot(matrix_B)) # Phi_1
    
    # acquire eigen vector with 2nd largest eigen values
    phi_1,psi_1 = second_sv_fn(matrix_B,sqrt_Py)
    
    gy = psi_1 / sqrt_Py
    
    import pdb
    M = matrix_B.T.dot(matrix_B)
    second_sv = np.sqrt(psi_1.T.dot(M).dot(psi_1))
    
    # gy = 
    #second_sv = 
    
    print('second_sv : {}'.format(second_sv))    
    plt.plot(np.arange(15), gy, c = 'r')
    plt.xlabel('y')
    plt.ylabel('gy')
    plt.show()

if __name__ == "__main__":
    main()
