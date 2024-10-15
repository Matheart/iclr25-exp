########### Loading uniform distribution on unit sphere in R^d with labels from N(0,1)-target function 0 #############

import torch
import numpy as np
import ipdb
def load_sphere(N_train,dim):

    trainX = np.random.normal(size=(N_train,dim)).astype('float32')
    trainX /= (trainX**2).sum(axis=1)[:,None]**.5
    
    testX = np.random.normal(size=(10000,dim)).astype('float32')
    testX /= (testX**2).sum(axis=1)[:,None]**.5

    trainY, testY = torch.tensor(1 * np.random.normal(size=(N_train,1))).long(), torch.tensor(np.zeros((10000,1))).long()
    
    
    trainX = torch.from_numpy(trainX).float()
    testX = torch.from_numpy(testX).float()

    return trainX,trainY,testX,testY

    