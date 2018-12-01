# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:14:36 2018

@author: XieTianwen
"""
'''
@author2: WANG Zifeng
'''


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

import pdb


x = np.linspace(-np.pi,np.pi,140).reshape(140,-1)
y = np.sin(x)

lr = 0.02     #set learning rate


def mean_square_loss(y_pre,y_true):         #define loss 
    loss = np.power(y_pre - y_true, 2).mean()*0.5
    loss_grad = (y_pre-y_true)/y_pre.shape[0]
    return loss , loss_grad           # return loss and loss_grad
    
class ReLU():                     # ReLu layer
    def __init__(self):
        pass
    def forward(self,inputs):
        '''
        your code
        '''
        # return  *
        return (np.abs(inputs)+inputs)/2

    def backward(self,inputs,grad_output):
        '''
        your code
        '''
        # return *
        
        d_nj = self.forward(inputs)
        d_nj[d_nj > 0] = 1.0

        delta = d_nj * grad_output  # [140,80] = [140,80] .* [140,80]
        
        return delta

        

class FC():
    def __init__(self,input_dim,output_dim):    # initilize weights
        self.W = np.random.randn(input_dim,output_dim)*1e-2
        self.b = np.zeros((1,output_dim))
                       
    def forward(self,inputs):          
        '''
        your codes
        '''
        # return *
        outputs = np.dot(inputs,self.W) + self.b
        return outputs
        
    
    def backward(self,inputs,grad_out):       # backpropagation , update weights in this step
        '''
        your codes
        '''
        #self.W -= lr * delt_W
        #self.b -= lr * delt_b
        # return *
        
        self.W -= lr * np.dot(inputs.T,grad_out)
        self.b -= lr * np.sum(grad_out,0)
        
        sum_delta_k = np.dot(grad_out,self.W.T)

        """
        for layer2:
            w := [80,1] = [80,140] dot [140,1]
            b := [1,1]  = reduce_sum([140,1],0) = [1,1]
        for layer1:
            w := [1,80] = [1,140] dot [140,80]
            b := [1,80] = reduce_sum([140,80],0) = [1,80]
        """
        
        return sum_delta_k



#  bulid the network      
layer1 = FC(1,80)
ac1 = ReLU()
out_layer = FC(80,1)

# count steps and save loss history
loss = 1
step = 0
l= []
while loss >= 1e-4 and step < 15000: # training 
            
    # forward     input x , through the network and get y_pre and loss_grad   
    
    '''
    your codes
    '''
    
    l1 = layer1.forward(x)
    ac_l1 = ac1.forward(l1)
    l2 = out_layer.forward(ac_l1)

    #backward   # backpropagation , update weights through loss_grad
    
    '''
    your codes
    '''
    loss , loss_grad = mean_square_loss(l2,y)
    print("{}/15000  loss: {}".format(step+1,loss))

    sum_delta_k = out_layer.backward(ac_l1,loss_grad)
    ac_delta = ac1.backward(l1,sum_delta_k)
    layer1.backward(x,ac_delta)
        
    step += 1
    l.append(loss)
    
    
    
# after training , plot the results
y_pre = l2
plt.plot(x,y,c='r',label='true_value')
plt.plot(x,y_pre,c='b',label='predict_value')
plt.legend()
plt.savefig('1.png')
plt.figure()
plt.plot(np.arange(0,len(l)), l )
plt.title('loss history')
plt.show() 
