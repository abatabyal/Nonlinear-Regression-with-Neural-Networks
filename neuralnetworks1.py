'''
This is contributed by Professor Charles Anderson, CSU
Neural network with one hidden layer.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNetwork(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units,
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh

 For nonlinear classification (prediction of integer valued class labels)
   net = NeuralNetworkClassifier(ni,nh,no)
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x 1 (integer class labels
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   classes,Y,Z = net.use(Xtest,allOutputs=True)  # classes is nSamples x 1
'''


import scaledconjugategradient as scg
# reload(gd)
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp  # to allow access to number of elapsed iterations
#import cPickle
import mlutils as ml

# def pickleLoad(filename):
#     with open(filename,'rb') as fp:
#         nnet = cPickle.load(fp)
#     nnet.iteration = mp.Value('i',0)
#     nnet.trained = mp.Value('b',False)
#     return nnet

class NeuralNetwork:
    def __init__(self,ni,nhs,no):
        try:
            nihs = [ni] + list(nhs)
        except:
            nihs = [ni] + [nhs]
            nhs = [nhs]
        self.Vs = [np.random.uniform(-0.1,0.1,size=(1+nihs[i],nihs[i+1])) for i in range(len(nihs)-1)]
        self.W = np.random.uniform(-0.1,0.1,size=(1+nhs[-1],no))
        # print [v.shape for v in self.Vs], self.W.shape
        self.ni,self.nhs,self.no = ni,nhs,no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.iteration = mp.Value('i',0)
        self.trained = mp.Value('b',False)
        self.reason = None
        self.errorTrace = None
        
    def getSize(self):
        return (self.ni,self.nhs,self.no)

    def getErrorTrace(self):
        return self.errorTrace
    
    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def train(self,X,T,
              nIterations=100,weightPrecision=0,errorPrecision=0,verbose=False):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self._standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
        T = self._standardizeT(T)

        # Local functions used by gradientDescent.scg()

        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            return 0.5 * np.mean((Y - T)**2)

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            delta = (Y - T) / (X.shape[0] * T.shape[1])
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace) - 1
        self.trained.value = True
        return self
    
    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        Y = self._unstandardizeT(Y)
        return (Y,Z[1:]) if allOutputs else Y

    def draw(self,inputNames = None, outputNames = None):
        ml.draw(self.Vs, self.W, inputNames, outputNames)

    def __repr__(self):
        str = 'NeuralNetwork({}, {}, {})'.format(self.ni,self.nhs,self.no)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,
                                                                                           self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str
            
    def _standardizeX(self,X):
        return (X - self.Xmeans) / self.Xstds
    def _unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def _standardizeT(self,T):
        return (T - self.Tmeans) / self.Tstds
    def _unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans

    def _forward_pass(self,X):
        Zprev = X
        Zs = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Zs.append(Zprev)
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        return Y, Zs

    def _backward_pass(self,delta,Z):
        dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
        dVs = []
        delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
        for Zi in range(len(self.nhs),0,-1):
            Vi = Zi - 1 # because X is first element of Z
            dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                             np.dot( Z[Zi-1].T, delta)))
            dVs.insert(0,dV)
            delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
        return dVs,dW

    def _pack(self,Vs,W):
        # r = np.hstack([V.flat for V in Vs] + [W.flat])
        # print 'pack',len(Vs), Vs[0].shape, W.shape,r.shape
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self,w):
        first = 0
        numInThisLayer = self.ni
        for i in range(len(self.Vs)):
            self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
            first += (numInThisLayer+1) * self.nhs[i]
            numInThisLayer = self.nhs[i]
        self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))

    def pickleDump(self,filename):
        # remove shared memory objects. Can't be pickled
        n = self.iteration.value
        t = self.trained.value
        self.iteration = None
        self.trained = None
        with open(filename,'wb') as fp:
        #    pickle.dump(self,fp)
            cPickle.dump(self,fp)
        self.iteration = mp.Value('i',n)
        self.trained = mp.Value('b',t)

class NeuralNetworkClassifier(NeuralNetwork):
    def __init__(self,ni,nhs,no):
        #super(NeuralNetworkClassifier,self).__init__(ni,nh,no)
        NeuralNetwork.__init__(self,ni,nhs,no-1)

    def _multinomialize(self,Y):
        # fix to avoid overflow
        mx = np.max(Y)
        expY = np.exp(Y-mx)
        denom = np.exp(-mx) + np.sum(expY,axis=1).reshape((-1,1))
        # Y = np.hstack((expY / denom, 1/denom))
        rowsHavingZeroDenom = denom == 0.0
        if np.sum(rowsHavingZeroDenom) > 0:
            Yshape = (expY.shape[0],expY.shape[1]+1)
            nClasses = Yshape[1]
            # add random values to result in random choice of class
            Y = np.ones(Yshape) * 1.0/nClasses + np.random.uniform(0,0.1,Yshape)
            Y /= np.sum(Y,1).reshape((-1,1))
        else:
            Y = np.hstack((expY / denom, np.exp(-mx)/denom))
        return Y

    def train(self,X,T,
                 nIterations=100,weightPrecision=0,errorPrecision=0,verbose=False):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self._standardizeX(X)

        self.classes = np.unique(T)
        if self.no != len(self.classes)-1:
            raise ValueError(" In NeuralNetworkClassifier, the number of outputs must be one less than\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetworkClassifier()." % (self.no, len(self.classes)))
        T = ml.makeIndicatorVars(T)

        # Local functions used by gradientDescent.scg()
        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            Y = self._multinomialize(Y)
            return -np.mean(T * np.log(Y))

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            Y = self._multinomialize(Y)
            delta = (Y[:,:-1] - T[:,:-1]) / (X.shape[0] * (T.shape[1]-1))
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace) - 1
        self.trained.value = True
        return self
    
    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        Y = self._multinomialize(Y)
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        return (classes,Y,Z[1:]) if allOutputs else classes


if __name__== "__main__":
    plt.ion()

    print( '\n------------------------------------------------------------')
    print( "Regression Example: Approximate f(x) = 1.5 + 0.6 x + 0.4 sin(x)")
    # print( '                    Neural net with 1 input, 5 hidden units, 1 output')
    nSamples = 10
    X = np.linspace(0,10,nSamples).reshape((-1,1))
    T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
    T[np.logical_and(X > 2, X < 3)] *= 3
    T[np.logical_and(X > 5, X < 7)] *= 3
    
    nSamples = 100
    Xtest = np.linspace(0,10,nSamples).reshape((-1,1)) + 10.0/nSamples/2
    Ttest = 1.5 + 0.6 * Xtest + 0.8 * np.sin(1.5*Xtest) + np.random.uniform(-2,2,size=(nSamples,1))
    Ttest[np.logical_and(Xtest > 2, Xtest < 3)] *= 3
    Ttest[np.logical_and(Xtest > 5,Xtest < 7)] *= 3

    # # nnet = NeuralNetwork(1,(5,4,3,2),1)
    # # nnet = NeuralNetwork(1,(10,2,10),1)
    # # nnet = NeuralNetwork(1,(5,5),1)
    nnet = NeuralNetwork(1,(3,3,3,3),1)
    
    nnet.train(X,T,errorPrecision=1.e-10,weightPrecision=1.e-10,nIterations=1000)
    print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
    Y = nnet.use(X)
    Ytest,Ztest = nnet.use(Xtest, allOutputs=True)
    print( "Final RMSE: train", np.sqrt(np.mean((Y-T)**2)),"test",np.sqrt(np.mean((Ytest-Ttest)**2)))

    # import time
    # t0 = time.time()
    # for i in range(100000):
    #     Ytest,Ztest = nnet.use(Xtest, allOutputs=True)
    # print( 'total time to make 100000 predictions:',time.time() - t0)
    
    # print( 'Inputs, Targets, Estimated Targets')
    # print( np.hstack((X,T,Y)))

    plt.figure(1)
    plt.clf()
    
    nHLayers = len(nnet.nhs)
    nPlotRows = 3 + nHLayers

    plt.subplot(nPlotRows,2,1)
    plt.plot(nnet.getErrorTrace())
    plt.xlabel('Iterations');
    plt.ylabel('RMSE')
    
    plt.title('Regression Example')
    plt.subplot(nPlotRows,2,3)
    plt.plot(X,T,'o-')
    plt.plot(X,Y,'o-')
    plt.text(8,12, 'Layer {}'.format(nHLayers+1))
    plt.legend(('Train Target','Train NN Output'),loc='lower right',
               prop={'size':9})
    plt.subplot(nPlotRows,2,5)
    plt.plot(Xtest,Ttest,'o-')
    plt.plot(Xtest,Ytest,'o-')
    plt.xlim(0,10)
    plt.text(8,12, 'Layer {}'.format(nHLayers+1))
    plt.legend(('Test Target','Test NN Output'),loc='lower right',
               prop={'size':9})
    colors = ('blue','green','red','black','cyan','orange')
    for i in range(nHLayers):
        layer = nHLayers-i-1
        plt.subplot(nPlotRows,2,i*2+7)
        plt.plot(Xtest,Ztest[layer]) #,color=colors[i])
        plt.xlim(0,10)
        plt.ylim(-1.1,1.1)
        plt.ylabel('Hidden Units')
        plt.text(8,0, 'Layer {}'.format(layer+1))

    plt.subplot(2,2,2)
    nnet.draw(['x'],['sine'])
    plt.draw()

    
    # Now train multiple nets to compare error for different numbers of hidden layers

    if False:  # make True to run multiple network experiment
        
        def experiment(hs,nReps,nIter,X,T,Xtest,Ytest):
            results = []
            for i in range(nReps):
                nnet = NeuralNetwork(1,hs,1)
                nnet.train(X,T,weightPrecision=0,errorPrecision=0,nIterations=nIter)
                # print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
                (Y,Z) = nnet.use(X, allOutputs=True)
                Ytest = nnet.use(Xtest)
                rmsetrain = np.sqrt(np.mean((Y-T)**2))
                rmsetest = np.sqrt(np.mean((Ytest-Ttest)**2))
                results.append([rmsetrain,rmsetest])
            return results

        plt.figure(2)
        plt.clf()
    
        results = []
        # hiddens [ [5]*i for i in range(1,6) ]
        hiddens = [[12], [6,6], [4,4,4], [3,3,3,3], [2,2,2,2,2,2],
                   [24], [12]*2, [8]*3, [6]*4, [4]*6, [3]*8, [2]*12]

        for hs in hiddens:
            r = experiment(hs,30,100,X,T,Xtest,Ttest)
            r = np.array(r)
            means = np.mean(r,axis=0)
            stds = np.std(r,axis=0)
            results.append([hs,means,stds])
            print( hs, means,stds)
        rmseMeans = np.array([x[1].tolist() for x in results])
        plt.clf()
        plt.plot(rmseMeans,'o-')
        ax = plt.gca()
        plt.xticks(range(len(hiddens)),[str(h) for h in hiddens])
        plt.setp(plt.xticks()[1], rotation=30)
        plt.ylabel('Mean RMSE')
        plt.xlabel('Network Architecture')

        
    print( '\n------------------------------------------------------------')
    print( "Classification Example: XOR, approximate f(x1,x2) = x1 xor x2")
    print( '                        Using neural net with 2 inputs, 3 hidden units, 2 outputs')
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    T = np.array([[1],[2],[2],[1]])
    nnet = NeuralNetworkClassifier(2,(4,),2)
    nnet.train(X,T,weightPrecision=1.e-10,errorPrecision=1.e-10,nIterations=100)
    print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
    (classes,y,Z) = nnet.use(X, allOutputs=True)
    

    print( 'X(x1,x2), Target Classses, Predicted Classes')
    print( np.hstack((X,T,classes)))

    print( "Hidden Outputs")
    print( Z)
    
    plt.figure(3)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(np.exp(-nnet.getErrorTrace()))
    plt.xlabel('Iterations');
    plt.ylabel('Likelihood')
    plt.title('Classification Example')
    plt.subplot(2,1,2)
    nnet.draw(['x1','x2'],['xor'])

