import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
# Neural network implementation in numpy 

class alvinNet:
    ''' double layered neural network in completely coded in numpy 
    start - input features (initial neurons)
    hidden - Number of neurons in hidden layer
    end - output layer
    
    '''
    def __init__(self, start ,hidden = 5 ,end = 1):
        self.start = start
        self.end = end
        self.hidden = hidden
    
    def ini_weights(self, seed = None):
        ''' Random initialization of weights. Bias initialized to 0  '''
        miniW = 0.1 # to keep the wights small
        np.random.seed(seed)
        self.W1 = np.random.randn(self.start, self.hidden) * miniW
        self.b1 = np.random.randn(1,self.hidden)
        # while addition the feature dim is columns 
        self.W2 = np.random.randn( self.hidden, self.end) * miniW
        self.b2 = np.random.randn(1,self.end)  

    
    def _sigmoid(self,x): # activation function
        return 1/(1+np.e**(-x))   
    
    def info(self):
        print(f"layer1 weights :\n{self.W1}")
        print(f"Bias :\n {self.b1}")
        print('-'*20)
        print(f"layer2 weights :\n{self.W2}")
        print(f"Bias : \n{self.b2}")
        print('-'*20)
    
    def forward(self,X):
        ''' X- input for a single sample the output is predicted'''
        
        # calculations for HIdden layer 
        Z1 = np.dot(X,self.W1) + self.b1 
#        print(Z1)
        self.A1 = np.tanh(Z1) # hidden layer neurons value
        
        # calculations for final layer
        Z2 = np.dot(self.A1,self.W2) + self.b2
        self.A2 = self._sigmoid(Z2)    # Final output prediction
        return self.A2
  
    def cross_entropy(self,Y,show = False):
        logprobs = np.multiply(np.log(self.A2), Y) + np.multiply((1 - Y), np.log(1 - self.A2))
        cost = -np.sum(logprobs) / Y.shape[0]
        return cost
            
    def back_prop(self,learning_rate = 0.1):
        
        m = self.Y.shape[0] # number of smaples 
        dZ2 = self.A2 - self.Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
#        print(" dW2 ",dW2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
#        print(" db2 ",db2)
        
        dZ1 = np.multiply(np.dot( dZ2,self.W2.T), 1 - np.power(self.A1, 2))
#        print(" dZ1 ",dZ1)
        dW1 = (1 / m) * np.dot(self.X.T, dZ1)
#        print(" dW1 ",dW1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # updating the weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def run(self,X,Y,epochs = 10,learning_rate = 0.1):
        '''
        X = input Data (samples, features)
        Y = output Data (samples , features)
        '''
        
        if self.start != X.shape[1]:
            print('-'*30)
            print(f"input dimensions don't match start >> {self.start} != X.shape[1] >>{X.shape[1]} ")
            print('-'*30)
            return
        
        if Y.shape[0] != X.shape[0]:
            print('-'*30)
            print(f" Samples are inequal " )
            print('-'*30)
            return
        
        # making the data accessible throughout the class
        
        self.Y = Y  
        # the actual dimension should be ( features, samples)  for output layer
        self.X = X
        
        for i in range(epochs):

            self.forward(self.X)
            c = self.cross_entropy(self.Y,True)
            self.back_prop(learning_rate)
            if i%100 == 0:
                print(f"cost : {c}")
        print("training completed")
        
        self.parameters = {"W1":self.W1,    \
                           "b1":self.b1,    \
                           "W2":self.W2,    \
                           "b2":self.b2,    \
                           "act1":np.tanh,  \
                           "act2":self._sigmoid, \
                           "info":f" start : {self.start} \n hidden : {self.hidden} \n \
                           end : {self.end} \n act1 : tanh \n act1 : sigmoid"}
    def predict(self,a):
    
        res = self.forward(a)
        return res > 0.5
        
        
    # Weights saving and loading
    
    def saveNN(self, loc = "./" , filename = 'model.pkl'):
        ''' 
        Save with the extension .pkl
        '''
        with open(loc+filename,'wb') as file:
            pkl.dump(self,file)
        print(f"Model Save Completed in loc : {loc+filename}")
    
    def saveWeights(self, loc = "./" , filename = 'weights.pkl'):
        ''' 
        Save with the extension .pkl
        '''
        with open(loc+filename,'wb') as file:
            pkl.dump(self.parameters,file)
        print(f"Weights Save Completed in loc : {loc+filename}")
        
    def loadWeights(self, loc = "./" , filename = 'weights.pkl'):
        ''' 
        Save with the extension .pkl
        '''
        with open(loc+filename,'rb') as file:
            w = pkl.load(file)
        
        self.W1 = w["W1"]
        self.b1 = w["b1"]
        self.W2 = w["W2"]
        self.b2 = w["b2"]
        
        print(f"Weights Loaded sucessfull ")
        self.info()
        
        
        
    def genANDData(self, samples = 100):
        '''
        Data generation for AND GATE testing.
        '''
        X = np.random.randint(0,2,(samples,2),'int')
        Y = np.array(X[:,0] & X[:,1]).reshape((-1,1))
        return (X, Y)
         

nn = alvinNet(2,5,1)
nn.ini_weights()
#nn.info()

nn.run(*nn.genANDData(100),epochs = 1000)

xtest ,ytest = nn.genANDData()

ypreds = nn.predict(xtest)
print(f" Accuracy Score : {accuracy_score(ytest,ypreds)*100} %")
nn.info()

# saving the weights
nn.saveNN()
nn.saveWeights() 
nn.loadWeights()



