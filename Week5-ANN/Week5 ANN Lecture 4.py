# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:45:12 2019

@author: Soundarya Ganesh
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:45:12 2019

@author: Soundarya Ganesh
"""
from matplotlib import pyplot as plt
import numpy as np
import csv
data=[]
with open('data1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data.append(row)
csvFile.close()
#particulars=data[0]
del data[0]
train_data=[]
#print(data)
for i in range (0,len(data)-5):
    train_data.append(data[i])
test_data=[]
for i in range(1,6):
    test_data.append(data[-i])
def Raf(x):               #Activation function
        return max(0.5*x,x)
def der_Raf(x):           #derivative of function
    if x<=0:
        return 0
    else:
        return 1

def train():
    w1 = np.random.randn()
    w2 = np.random.randn()      #weights
    b  = np.random.randn()      #biase
    iterations = 100
    alpha = 0.4
    costs = []
    for i in range(iterations):
        ri = np.random.randint(len(train_data))
        point = train_data[ri]
        z = float(point[0]) * w1 + float(point[1]) * w2 + b
        #input to 2 input neurons
        
        pred = Raf(z) # networks prediction
        target = point[2]
        cost = np.square(float(pred) - float(target))
        if i % 100 == 0:
            c = 0
            for j in range(len(train_data)):
                p = train_data[j]
                p_pred = Raf(w1 * float(p[0]) + w2 * float(p[1]) + b)
                c += np.square(float(p_pred) - float(p[2]))
            costs.append(c)             #it should decrease in every 100 values or iterations
        dcost_dpred = 2 * (float(pred) - float(target))
        dpred_dz = der_Raf(z)
        
        dz_dw1 = float(point[0])
        dz_dw2 = float(point[1])
        dz_db = 1
        
        dcost_dz = dcost_dpred * dpred_dz
        
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db
        
        w1 = w1 - alpha * dcost_dw1
        w2 = w2 - alpha * dcost_dw2
        b = b - alpha * dcost_db
        
    return costs, w1, w2, b
costs, w1, w2, b = train()
for i in range (0, len(test_data)):
    X= w1 * float(test_data[i][0]) + w2 * float(test_data[i][1]) + b
    pred = Raf(X)
print(pred)
print(test_data[i][2])