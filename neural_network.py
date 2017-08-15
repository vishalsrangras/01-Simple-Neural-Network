import numpy as np

class Neural_Network(object):

    #The init method in python is like a constructor in Java
    def __init__(self):
        
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector or matrix
        return 1/(1+np.exp(-z))

if __name__ == "__main__":
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75],[82],[93]), dtype=float)
    print("Input (X): ")
    print(X)
    print("Output (y):")
    print(y)
    X = X/np.amax(X, axis=0)
    y = y/100 
    NN = Neural_Network()
    yHat = NN.forward(X)
    print("Estimated Output (yHat) :")
    print(yHat)
    print("Normalized Real Output (y) :")
    print(y)
