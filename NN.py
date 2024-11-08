import math
import numpy as np
import pandas as pd
from helper import helper as h
global data,labels,mem

data = np.array(pd.read_csv("Student_data.csv"))
data,labels = h.data_changer(data,1,13),data[:,-1]

mem = h.init_mem(data,labels)

class NN():
  def __init__(self):
    self.check = False
    self.params,self.prev_params = self.init_params(),self.init_prevData()
    for param,prev in zip(self.params,self.prev_params):
      np.longdouble(param)
      np.longdouble(prev)
    self.init_data()

  def init_data(self):
    # for i in range(1):
    global data
    #np.random.shuffle(data)
    self.training_data = []
    self.training_labels = []
    self.test_data = []
    self.test_labels = []
    freq_arr = np.zeros(5)
    for datum in data:
      val = mem[tuple(datum)]
      if freq_arr[int(val)] < 100:
        self.training_data.append(np.array(datum))
        self.training_labels.append(val)
        freq_arr[int(val)] += 1
      else:
        self.test_data.append(np.array(datum))
        self.test_labels.append(val)

  def init_params(self):
    self.W0 = (np.random.randn(50,13))
    self.B0 = (np.random.randn(50,1))
    self.W1 = (np.random.randn(20,50))
    self.B1 = (np.random.randn(20,1))
    self.W2 = (np.random.randn(10,20))
    self.B2 = (np.random.randn(10,1))
    self.W3 = (np.random.randn(5,10))
    self.B3 = (np.random.randn(5,1))
    return self.W0,self.B0,self.W1,self.B1,self.W2,self.B2,self.W3,self.B3

  def init_prevData(self):
    self.prev_W0 = np.zeros((50,13))
    self.prev_B0 = np.zeros((50,1))
    self.prev_W1 = np.zeros((20,50))
    self.prev_B1 = np.zeros((20,1))
    self.prev_W2 = np.zeros((10,20))
    self.prev_B2 = np.zeros((10,1))
    self.prev_W3 = np.zeros((5,10))
    self.prev_B3 = np.zeros((5,1))
    return self.prev_W0,self.prev_B0,self.prev_W1,self.prev_B1,self.prev_W2,self.prev_B2,self.prev_W3,self.B3


  def forward_prop(self,I,max = False):
    if max:
      self.W0,self.B0,self.W1,self.B1,self.W2,self.B2,self.W3,self.B3 = self.max_params
    Z0 = np.dot(self.W0,I) + self.B0
    A0 = np.array([h.Leaky(zi[0]) for zi in Z0]).reshape(50,1)
    Z1 = np.dot(self.W1,A0) + self.B1
    A1 = np.array([h.Leaky(Zi[0]) for Zi in Z1]).reshape(20,1)
    Z2 = np.dot(self.W2,A1) + self.B2
    A2 = np.array([h.Leaky(Zi[0]) for Zi in Z2]).reshape(10,1)
    Z3 = np.dot(self.W3,A2) + self.B3
    A3 = np.array([h.softmax(Zi[0],Z3) for Zi in Z3]).reshape(5,1)
    return A0,A1,A2,A3,Z0,Z1,Z2

  def back_prop(self,A0,A1,A2,A3,Z0,Z1,Z2,I,one_hot):
    dZ3 = h.dCostSoft(A3,one_hot).reshape(5,1)
    dW3 = h.clip(np.dot(dZ3,A2.T))
    dB3 = h.clip(dZ3)
    temp = np.dot(self.W3.T,dZ3)
    temp1 = h.hamard(h.dLeaky,Z2).reshape(10,1)
    dZ2 = h.hamard(temp1,temp).reshape(10,1)
    dW2 = h.clip(np.dot(dZ2,A1.T))
    dB2 = h.clip(dZ2)
    temp = np.dot(self.W2.T,dZ2)
    temp1 = h.hamard(h.dLeaky,Z1).reshape(20,1)
    dZ1 = h.hamard(temp1,temp).reshape(20,1)
    dW1 = h.clip(np.dot(dZ1,A0.T))
    dB1 = h.clip(dZ1)
    temp = np.dot(self.W1.T,dZ1)
    temp1 = h.hamard(h.dLeaky,Z0).reshape(50,1)
    dZ0 = h.hamard(temp1,temp).reshape(50,1)
    dW0 = h.clip(np.dot(dZ0,I.T))
    dB0 = h.clip(dZ0)
    return dW0,dB0,dW1,dB1,dW2,dB2,dW3,dB3

  def update_max(self):
    self.max_params = []
    for param in self.params:
      self.max_params.append(np.copy(param))

  def get_accuracy(self):
    c,t = 0,0
    for I,ans in zip(self.test_data,self.test_labels):
      t+=1
      I = I.reshape(13,1)
      A0,A1,A2,A3,Z0,Z1,Z2 = self.forward_prop(I,max=True)
      predicted = h.getMax(A3)
      if predicted == ans:
        c+=1
    return c/t


  def train(self,epochs):
    lr = .1
    lrb = .1
    a = .05
    b = .05
    max_accuracy = 0
    for epoch in range(epochs):
      np.random.shuffle(self.training_data)
      c = 0
      t = 0
      for I in self.training_data:
        ans = mem[tuple(I)]
        I = I.reshape(13,1)
        A0,A1,A2,A3,Z0,Z1,Z2 = self.forward_prop(I)
        predicted = h.getMax(A3)
        one_h = h.one_hot(ans)
        t+=1
        if predicted == ans:
          c+=1
        accuracy = c/t
        cur_gradients = self.back_prop(A0,A1,A2,A3,Z0,Z1,Z2,I,one_h)
        dW0,dB0,dW1,dB1,dW2,dB2,dW3,dB3 = [lr*cur + a*prev for cur,prev in zip(cur_gradients,self.prev_params)]
        self.W0 -= dW0
        self.B0 -= dB0/10
        self.W1 -= dW1
        self.B1 -= dB1/10
        self.W2 -= dW2
        self.B2 -= dB2/10
        self.W3 -= dW3
        self.B3 -= dB3/10
        self.prev_params = [dW0,dB0,dW1,dB1,dW2,dB2,dW3,dB3]
      if accuracy > max_accuracy:
        max_accuracy = accuracy
        self.update_max()
      if accuracy>.8:
        return
      print(accuracy)



