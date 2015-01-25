'''
This is contributed by Professor Charles Anderson, CSU
'''
import pdb
import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Machine Learning Utilities. 
#
#  makeIndicatorVars
#  segmentize
#  confusionMatrix
#  printConfusionMatrix
#  partition
#  draw (a neural network)

######################################################################


def makeIndicatorVars(T):
    """ Assumes argument is N x 1, N samples each being integer class label """
    return (T == np.unique(T)).astype(int)


# Build matrix of labeled time windows, each including nlags consecutive
# samples, with channels concatenated.  Keep only segments whose nLags
# targets are the same. Assumes target is in last column of data.
def segmentize(data,nLags):
    nSamples,nChannels = data.shape
    nSegments = nSamples-nLags+1
    # print 'segmentize, nSamples',nSamples,'nChannels',nChannels,'nSegments',nSegments
    # print 'data.shape',data.shape,'nLags',nLags
    segments = np.zeros((nSegments, (nChannels-1) * nLags + 1))
    k = 0
    for i in range(nSegments):
        targets = data[i:i+nLags,-1]
        # In previous versions had done this...
        #    to throw away beginning and end of each trial, and between trials
        #       if targets[0] != 0 and targets[0] != 32: 
        #            # keep this row
        if (np.all(targets[0] == targets)):
            allButTarget = data[i:i+nLags,:-1]
            # .T to keep all from one channel together
            segments[k,:-1] = allButTarget.flat
            segments[k,-1] = targets[0]
            k += 1
    return segments[:k,:]

def confusionMatrixOld(actual,predicted,classes):
    nc = len(classes)
    confmat = np.zeros((nc,nc))
    for ri in range(nc):
        trues = actual==classes[ri]
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predicted[trues] == classes[ci]) / float(np.sum(trues))
    return confmat


def confusionMatrix(actual,predicted,classes,probabilities=None,probabilityThreshold=None):
    nc = len(classes)
    if probabilities is not None:
        predictedClassIndices = np.zeros(predicted.shape,dtype=np.int)
        for i,cl in enumerate(classes):
            predictedClassIndices[predicted == cl] = i
        probabilities = probabilities[np.arange(probabilities.shape[0]), predictedClassIndices.squeeze()]
    confmat = np.zeros((nc,nc+2)) # for samples above threshold this class and samples this class
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        if probabilities is None:
            keep = trues
            predictedThisClassAboveThreshold = predictedThisClass
        else:
            keep = probabilities[trues] >= probabilityThreshold
            predictedThisClassAboveThreshold = predictedThisClass[keep]
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
        confmat[ri,nc] = np.sum(keep)
        confmat[ri,nc+1] = np.sum(trues)
    return confmat

def printConfusionMatrix(confmat,classes):
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('%s' % '------'*len(classes))
    for i,t in enumerate(classes):
        print('%2d |' % (t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('%5.1f' % (100*confmat[i,i1]), end='')
        print('   (%d / %d)' % (int(confmat[i,len(classes)]), int(confmat[i,len(classes)+1])))

def partition(X,T,trainPercent,classification=False):
    """Usage: Xtrain,Train,Xtest,Ttest = partition(X,T,80,classification=True)
    or Xtrain,Train,Xtest,Ttest = partition(X,T,80,classification=False)
      X is nSamples x nFeatures.
      If classification=True, T is target class as integer. Data partitioned
        according to class proportions.
        """
    if classification == 1:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        print(classes)
        trainrows = []
        testrows = []
        for c in classes:
            cRows = np.where(T == c)[0]
            # np.random.shuffle(cRows) # randomly reorder, in place
            firstTestRow = int(0.5+len(cRows) * trainPercent/100.0)
            trainrows += list(cRows[:firstTestRow])
            testrows += list(cRows[firstTestRow:])
        trainrows.sort()
        testrows.sort()
        trainrows = np.array(trainrows)
        testrows = np.array(testrows)
        Xtrain = X[trainrows,:]
        Ttrain = T[trainrows,:]
        Xtest = X[testrows,:]
        Ttest = T[testrows,:]
    else:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        firstTestRow = int(0.5 + n*trainPercent/100.0)
        Xtrain = X[:firstTestRow,:]
        Ttrain = T[:firstTestRow,:]
        Xtest =  X[firstTestRow:,:]
        Ttest = T[firstTestRow:,:]
    return Xtrain,Ttrain,Xtest,Ttest

######################################################################
# Associated with  neuralnetworks.py
# Draw a neural network with weights in each layer as a matrix
######################################################################

def draw(VsArg,WArg, inputNames = None, outputNames = None, gray = False):
    def isOdd(x):
        return x % 2 != 0

    W = VsArg + [WArg]
    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len,inputNames))/4.0 if inputNames else 1
    ylim = 0
    
    for li in range(nLayers):
        ni,no = W[li].shape  #no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni,no = W[nLayers-1].shape  #no means number outputs this layer
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len,outputNames))/4.0)

    ax = plt.gca()

    x0 = 1
    y0 = 0 # to allow for constant input to first layer
    # First Layer
    if inputNames:
        # addx = max(map(len,inputNames))*0.1
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0-len(n)*0.2, y, n)
            x0 = max([1,max(map(len,inputNames))/4.0])

    for li in range(nLayers):
        Wi = W[li]
        ni,no = Wi.shape
        if not isOdd(li):
            # Odd layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1')
            for li in range(ni):
                ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
            # output lines
            for li in range(no):
                ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            ax.scatter(xs,ys,marker='v',s=1000,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(no)+ x0+0.5
            ys = np.arange(ni)+ y0 + 0.5
            aWi = abs(Wi)
            aWi = aWi / np.max(aWi) * 20 #50
            coords = np.meshgrid(xs,ys)
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            y0 += ni + 1
            x0 += -1 ## shift for next layer's constant input
        else:
            # Even layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1')
            # input lines
            for li in range(ni):
                ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
            # output lines
            for li in range(no):
                ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5]*no)
            ys = y0 + 0.5 + np.arange(no)
            ax.scatter(xs,ys,marker='>',s=1000,c='gray')
            # weights
            Wiflat = Wi.T.flatten()
            if gray:
                colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wiflat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs,ys)
            aWi = abs(Wiflat)
            aWi = aWi / np.max(aWi) * 20 # 50
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            x0 += ni + 1
            y0 -= 1 ##shift to allow for next layer's constant input

    # Last layer output labels 
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n)
    ax.axis([0,xlim, ylim,0])
    ax.axis('off')
