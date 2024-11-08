import math
import numpy as np
class helper:
  @staticmethod
  def data_changer(data,start,end):
    start,end = start,end
    m,n = data.shape
    new_data = np.zeros((m,(end+1)-start))
    for val in range(start,end+1):
      new_data[:,val-start] = data[:,val]
    return new_data
  
  @staticmethod
  def generate_similar(I):
    newData = []
    for val in I:
      range = (1/100)*val
      newVal = val + range*(np.random.rand()-.5)
      newData.append(newVal)
    return newData

  @staticmethod
  def expand_data(data,labels):
    new_data,new_labels = [],[]
    for datum, label in zip(data,labels):
      for i in range(1):
        new_data.append(helper.generate_similar(datum))
        new_labels.append(label)
    return np.array(new_data),np.array(new_labels)

  @staticmethod
  def init_mem(data,labels):
    mem = {}
    for datum,label in zip(data,labels):
      mem[tuple(datum)] = label
    return mem

  @staticmethod
  def Leaky(value):
    a = .1
    b = 1e-3
    if value>=0:
      return a*value
    return b * value

  @staticmethod
  def dLeaky(value):
    if value>=0:
      return .1
    return 1e-3

  @staticmethod
  def ReLu(value):
    return max(0,value)

  @staticmethod
  def dRelu(value):
    if max(0,value):
      return 1
    return 0

  @staticmethod
  def logReLu(value):
    if value > 0:
      return math.log(value)
    return 0

  @staticmethod
  def dLogReLu(value):
    if value > 0:
      return 1/value
    return 0

  @staticmethod
  def softmax(x,ar):
    total = np.float128(sum([math.e**z for z in ar]))
    return math.e**x/total

  @staticmethod
  def cost(A3,one_hot):
    return np.sum(-one_hot*np.log(A3))

  @staticmethod
  def dCostSoft(A3,one_hot):
    dZ3 = np.longdouble(A3-one_hot)
    return dZ3

  @staticmethod
  def getMax(arr):
    y = arr.reshape(1,5)
    return np.where(y[0] == max(y[0]))[0][0]

  @staticmethod
  def myTanh(value):
    return math.tanh(value/10)

  @staticmethod
  def dTanh(value):
    return 1-(math.tanh(value)**2)

  @staticmethod
  def hamard(dF,arr):
    m,n = np.shape(arr)
    newArr = np.zeros((m,n))
    for i in range(m):
      for j in range(n):
        if callable(dF):
          newArr[i][j] = dF(arr[i][j])
        else:
          newArr[i][j] = dF[i][j] * arr[i][j]
    return newArr

  @staticmethod
  def dCos(value):
    return -math.sin(value)

  @staticmethod
  def one_hot(v):
    ar = np.zeros(5)
    ar[int(v)] = 1
    return ar.reshape(5,1)

  @staticmethod
  def clipper(val):
    if val>1:
      return 1
    if val<-1:
      return -1
    return val

  @staticmethod
  def clip(ar):
    m,n=np.shape(ar)
    newArr = np.zeros((m,n))
    for i in range(m):
      for j in range(n):
        newArr[i][j] = np.longdouble(helper.clipper(ar[i][j]))
    return newArr






