import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import neuralnetworks1 as nn

!head no2.dat

import pandas
d = pandas.read_csv('no2.dat',delim_whitespace=True,header=None)

df = d.iloc[:,:].values
df.shape

Xnames = ['log of the number of cars per hour', 'temperature 2 mtr above ground (degree C)', 'wind speed (m/s)', 'temperature diff b/w 25 and 2 mtrs above ground (degree C)', 'wind direction (degrees between 0 and 360)','hour of day', 'day number from October 1. 2001']
Tnames = 'concentration of NO2'

X = df[:, 1:] 
T = df[:,0:1]
plt.figure(figsize=(10,10))
for c in range(X.shape[1]):
    plt.subplot(3,3, c+1)
    plt.plot(X[:,c], T, 'o', alpha=0.5)
    plt.ylabel(Tnames)
    plt.xlabel(Xnames[c])

def partition_train_validate_test(X,T, fractions, network_shapes, nRepetitions):
    trainFraction = fractions[0]
    if len(fractions) == 2:
        # Skip the validation step
        validateFraction = 0
        testFraction = fractions[1]
    else:
        validateFraction = fractions[1]
        testFraction = fractions[2]
        
    n = X.shape[0]
    nTrain = round(trainFraction * n)
    nValidate = round(validateFraction * n)
    nTest = n - nTrain - nValidate

    rowIndices = np.arange(n)

    results = []  # will contain nHiddenUnits, RMSEtrain, RMSEvalidate, RMSEtest

    for nHiddens in network_shapes:

        print('Working with hidden layers =', nHiddens)
    
        for rep in range(nRepetitions):

            # Random arrangement of row indices
            np.random.shuffle(rowIndices)
    
            # Assign Xtrain and Train. Remove columns from Xtrain that contain constant values.
            Xtrain = X[rowIndices[:nTrain],:]
            Ttrain = T[rowIndices[:nTrain],:]

            if nValidate > 0:
                # Assign Xvalidate and Tvalidate
                Xvalidate = X[rowIndices[nTrain:nTrain+nValidate],:]
                Tvalidate = T[rowIndices[nTrain:nTrain+nValidate],:]

            # Assign Xtest and Ttest
            Xtest = X[rowIndices[nTrain+nValidate:],:]
            Ttest = T[rowIndices[nTrain+nValidate:],:]
        
            # build the model and test it
            nnet = nn.NeuralNetwork(Xtrain.shape[1],nHiddens,1)
            nnet.train(Xtrain,Ttrain,nIterations=500)
            Ptrain = nnet.use(Xtrain)
            if nValidate > 0:
                Pvalidate = nnet.use(Xvalidate)
            Ptest,Ztest = nnet.use(Xtest, allOutputs=True)

            # Use the model to predict and calculate errors.
            RMSEtrain = rmse(Ptrain,Ttrain)
            if nValidate > 0:
                RMSEvalidate = rmse(Pvalidate,Tvalidate)
            RMSEtest = rmse(Ptest,Ttest)

            if nValidate > 0:
                results.append([nHiddens, RMSEtrain, RMSEvalidate, RMSEtest])
            else:
                results.append([nHiddens, RMSEtrain,RMSEtest])                
    
    return np.array(results), Xtrain, Ptrain, Ttrain, Ptest, Ttest, Xtest, nnet, Ztest

def rmse(P,T):
    return np.sqrt(np.mean((P-T)**2))

results,_,_,_,_,_,_,_,_= partition_train_validate_test(X,T, (0.6,0.2,0.2), (1,2,5,10,20,40,50),40)
