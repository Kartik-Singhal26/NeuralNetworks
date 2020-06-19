# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:50:34 2020

@author: kartik

This file contains the following:
 
A. class Activation_functions()

It contains few of the below-mentioned activation functions and their derivatives.
User can try different activation functions as per their requirement. 
    1. Loss function: Cross Entropy
    2. Activation Functions:
    [For Hidden Layers]
        a. Relu z
        b. Sigmoid z
        c. Tanh z
        d. Arctan z
   [For Output Layer]
        a. Softmax Function

B. class NeuralNetwork_2HiddenLayers()

It is a 2 Hidden Layer Neural Network

C. class NeuralNetwork_3HiddenLayers()

It is a 3 Hidden Layer Neural Network 
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt


class Activation_functions():
    
    #Activation Functions [For Hidden Layers]
    def sigmoid(z):
        #It takes input as the layer vector z, where z = w*x + b
        #It returns the sigmoid function value, squashes value between 0 and 1
        z = z - np.max(z)
        return 1/(1 + np.exp(-z))
    
    def Relu(z):
        #It performs the threshold operation
        return np.maximum(0,z)
    
    def TanH(z):
        #It performs the tanh function and returns a value between -1 and 1
        z = z- np.max(z)
        return (2/(1 + np.exp(-2*z))) - 1
    
    def arcTan(z):
        z = z - np.max(z)
        return m.atan(z)
    
    #Activation Function [For Output Layer]
    def softmax(z):
        #This is the softmax function. It converts logits into probabilities
        num = np.exp(z - np.max(z))                                             # For stability subtract maximum value 
        dem = np.sum(num)                                                       # of array from each array element
        return num/dem
        
    #Cross Entropy Loss Function
    def crossentropy(Y,Y_predicted):
        n = len(Y)
        return (-1/n)*np.sum(Y*np.log(Y_predicted))
       
    #All derivatives required for backpropogation are defined
    
    def deriv_Sigmoid(s):
        #Input is Sigmoid(z)       
        return s*(1-s)
    
    def deriv_Relu(z):
        z[z > 0] = 1
        z[z <= 0] = 0
        return z
    
    def deriv_TanH(s):
        #Input is s = Tanh(z)
        return 1 - s**2
    
    def deriv_ArcTan(z):
        return 1/(z**2 + 1)
    
    def deriv_CE_Softmax(Y, Y_predicted):
        length = Y.shape[0]
        return (Y_predicted - Y)/length
    
    
 # A 2 Hidden Layer Neural Network   

class NeuralNetwork_2HiddenLayers():
    
    def __init__(self, X, Y, layers, learning_rate, epoches):
        self.params = {}
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.loss = []
        self.sample_Size = None
        self.layers = layers
        self.X = None
        self.Y = None
        
    def initial_weights(self, X, Y):
        #Initialize Weights from random distribution
        input_layer = X.shape[1]
        output_layer = Y.shape[1]
        
        # Seed the random number generator
        np.random.seed(1) 
        
        self.params["W1"] = np.random.randn(input_layer, self.layers[0])        #Hidden Layer1: W1 = (6,64)
        self.params['B1'] = np.random.randn(self.layers[0],)                    #HiddenLayer1: B1 = (64,)
        
        self.params['W2'] = np.random.randn(self.layers[0], self.layers[1])     #Hidden Layer2: W2 = (64,32)
        self.params['B2'] = np.random.randn(self.layers[1],)                    #Hidden Layer2: B2 = (32,)
        
        self.params['W3'] = np.random.randn(self.layers[1], output_layer)       #Output Layer: W2 = (32,5)
        self.params['B3'] = np.random.randn(output_layer,)                      #Output Layer : B3 = (5,) 
            
    #Forward Propogation: Training
    def forward_prop(self, X):
        Z_Layer1 = X.dot(self.params['W1']) + self.params['B1']
        self.params['Z_Layer1'] = Z_Layer1
        
        A1 = Activation_functions.sigmoid(Z_Layer1)
        self.params['A1'] = A1
        
        Z_Layer2 =  A1.dot(self.params['W2']) + self.params['B2']
        self.params['Z_Layer2'] = Z_Layer2
        
        A2 = Activation_functions.sigmoid(Z_Layer2)
        self.params['A2'] = A2
        
        Z_Layer3 =  A2.dot(self.params['W3']) + self.params['B3']
        self.params['Z_Layer3'] = Z_Layer3
        
        A3 = Activation_functions.softmax(Z_Layer3)                            #Predicted Value
        self.params['A3'] = A3
        
        return A3 
    
    #Back Propogation 
    def back_prop(self):
        #Calculate Loss
        calculated_Loss = Activation_functions.crossentropy(self.Y, self.params['A3'])
        
        #Calculate Derivatives
        A3_wrt_Output = Activation_functions.deriv_CE_Softmax(self.Y, self.params['A3'])
        
        z2_wrt_W3 =  A3_wrt_Output.dot(self.params['W3'].T)
        z2_wrt_A2 = z2_wrt_W3 * Activation_functions.deriv_Sigmoid(self.params['A2'])
        
        z1_wrt_W2 = z2_wrt_A2.dot(self.params['W2'].T)
        z1_wrt_A1 = z1_wrt_W2 * Activation_functions.deriv_Sigmoid(self.params['A1'])
        
        dLoss_W1 = self.X.T.dot(z1_wrt_A1)
        dLoss_W2 = self.params['A1'].T.dot(z2_wrt_W3)
        dLoss_W3 = self.params['A2'].T.dot(A3_wrt_Output)
        
        dLoss_B1 = np.sum(z1_wrt_A1)
        dLoss_B2 = np.sum(z2_wrt_A2)
        dLoss_B3 = np.sum(A3_wrt_Output)
        
        #Update weights and biases
        self.params['W1'] = self.params['W1'] - self.learning_rate * dLoss_W1 
        self.params['W2'] = self.params['W2'] - self.learning_rate * dLoss_W2 
        self.params['W3'] = self.params['W3'] - self.learning_rate * dLoss_W3 
        
        self.params['B1'] = self.params['B1'] - self.learning_rate * dLoss_B1
        self.params['B2'] = self.params['B2'] - self.learning_rate * dLoss_B2
        self.params['B3'] = self.params['B3'] - self.learning_rate * dLoss_B3
        
        return calculated_Loss
       
    #Training Function
    def Training(self, Xtrain, Ytrain, learning_rate, epoches):
        self.X = Xtrain
        self.Y = Ytrain 
        self.initial_weights(Xtrain, Ytrain)
        
        for i in range(1,self.epoches+1):
            self.forward_prop(self.X)
            loss = self.back_prop()
            self.loss.append(loss)
            
            print('Epoch: ',i, ', Calculated Loss: ', loss)
            
    #Prediction  
    def Predict(self, Data):
        self.X = Data
        value = self.forward_prop(self.X)                                             #Predicted Value
        
        return np.round(value.argmax())
    
    #Determine accuracy of training
    def accuracy(self, train, test):
        accuracy = 0
        for tr,te in zip(train,test):
            pred = self.Predict(tr)
            if pred == np.argmax(te):
                accuracy +=1
        return (accuracy/len(train))*100

    #Plot the neural network loss
    def plotNeuralNet(self,Train,Test):
        #Text Position
        x = 50
        y = (self.loss[-1] + self.loss[0])/2
        #Plot
        plt.plot(self.loss)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Calculated Loss')
        plt.title('Loss curve of training the Neural Network')
        plt.text(x,y,'Training Accuracy: {}'.format(round(Train,2)), fontsize=10)
        plt.text(x,(y + 0.1),'Testing Accuracy: {}'.format(round(Test,2)), fontsize=10)
        plt.show()
        
# A 3 Hidden Layer Neural Network

class NeuralNetwork_3HiddenLayers():
    
    def __init__(self, X, Y, layers, learning_rate, epoches):
        self.params = {}
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.loss = []
        self.sample_Size = None
        self.layers = layers
        self.X = None
        self.Y = None
        
    def initial_weights(self, X, Y):
        #Initialize Weights from random distribution
        input_layer = X.shape[1]
        output_layer = Y.shape[1]
        
        # Seed the random number generator
        np.random.seed(1) 
        
        self.params["W1"] = np.random.randn(input_layer, self.layers[0])       #Hidden Layer1: W1 
        self.params['B1'] = np.random.randn(self.layers[0],)                   #Hidden Layer1: B1
        
        self.params['W2'] = np.random.randn(self.layers[0], self.layers[1])    #Hidden Layer2: W2
        self.params['B2'] = np.random.randn(self.layers[1],)                   #Hidden Layer2: B2
        
        self.params['W3'] = np.random.randn(self.layers[1], self.layers[2])    #Hidden Layer3: W3
        self.params['B3'] = np.random.randn(self.layers[2],)                   #Hidden Layer3: B3
        
        self.params['W4'] = np.random.randn(self.layers[2], output_layer)      #Output Layer: W4,B4
        self.params['B4'] = np.random.randn(output_layer,) 
                   
    #Forward Propogation: Training
    def forward_prop(self, X):
        Z_Layer1 = X.dot(self.params['W1']) + self.params['B1']
        self.params['Z_Layer1'] = Z_Layer1
        
        A1 = Activation_functions.sigmoid(Z_Layer1)
        self.params['A1'] = A1
        
        Z_Layer2 =  A1.dot(self.params['W2']) + self.params['B2']
        self.params['Z_Layer2'] = Z_Layer2
        
        A2 = Activation_functions.sigmoid(Z_Layer2)
        self.params['A2'] = A2
        
        Z_Layer3 =  A2.dot(self.params['W3']) + self.params['B3']
        self.params['Z_Layer3'] = Z_Layer3
        
        A3 = Activation_functions.sigmoid(Z_Layer3)                                            
        self.params['A3'] = A3
        
        Z_Layer4 =  A3.dot(self.params['W4']) + self.params['B4']
        self.params['Z_Layer4'] = Z_Layer4
        
        A4 = Activation_functions.softmax(Z_Layer4)                            #Predicted Value
        self.params['A4'] = A4
        
        return A4
    
    #Back Propogation 
    def back_prop(self):
        #Calculate Loss
        calculated_Loss = Activation_functions.crossentropy(self.Y, self.params['A4'])
        
        #Calculate Derivatives
        A4_wrt_Output = Activation_functions.deriv_CE_Softmax(self.Y, self.params['A4'])
        
        z3_wrt_W4 =  A4_wrt_Output.dot(self.params['W4'].T)
        z3_wrt_A3 = z3_wrt_W4 * Activation_functions.deriv_Sigmoid(self.params['A3'])
        
        z2_wrt_W3 =  z3_wrt_A3.dot(self.params['W3'].T)
        z2_wrt_A2 = z2_wrt_W3 * Activation_functions.deriv_Sigmoid(self.params['A2'])
        
        z1_wrt_W2 = z2_wrt_A2.dot(self.params['W2'].T)
        z1_wrt_A1 = z1_wrt_W2 * Activation_functions.deriv_Sigmoid(self.params['A1'])
        
        dLoss_W1 = self.X.T.dot(z1_wrt_A1)
        dLoss_W2 = self.params['A1'].T.dot(z2_wrt_W3)
        dLoss_W3 = self.params['A2'].T.dot(z3_wrt_W4)
        dLoss_W4 = self.params['A3'].T.dot(A4_wrt_Output)
        
        dLoss_B1 = np.sum(z1_wrt_A1)
        dLoss_B2 = np.sum(z2_wrt_A2)
        dLoss_B3 = np.sum(z3_wrt_A3)
        dLoss_B4 = np.sum(A4_wrt_Output)
        
        #Update weights and biases
        self.params['W1'] = self.params['W1'] - self.learning_rate * dLoss_W1 
        self.params['W2'] = self.params['W2'] - self.learning_rate * dLoss_W2 
        self.params['W3'] = self.params['W3'] - self.learning_rate * dLoss_W3 
        self.params['W4'] = self.params['W4'] - self.learning_rate * dLoss_W4 
        
        self.params['B1'] = self.params['B1'] - self.learning_rate * dLoss_B1
        self.params['B2'] = self.params['B2'] - self.learning_rate * dLoss_B2
        self.params['B3'] = self.params['B3'] - self.learning_rate * dLoss_B3
        self.params['B4'] = self.params['B4'] - self.learning_rate * dLoss_B4
        
        return calculated_Loss
       
    #Training Function
    def Training(self, Xtrain, Ytrain, learning_rate, epoches):
        self.X = Xtrain
        self.Y = Ytrain 
        self.initial_weights(Xtrain, Ytrain)
        
        for i in range(1,self.epoches+1):
            self.forward_prop(self.X)
            loss = self.back_prop()
            self.loss.append(loss)
            
            print('Epoch: ',i, ', Calculated Loss: ', loss)
    
    #Prediction  
    def Predict(self, Data):
        self.X = Data
        value = self.forward_prop(self.X)                                             #Predicted Value
        
        return np.round(value.argmax())
    
    #Determine accuracy of training
    def accuracy(self, train, test):
        accuracy = 0
        for tr,te in zip(train,test):
            pred = self.Predict(tr)
            if pred == np.argmax(te):
                accuracy +=1
        return (accuracy/len(train))*100

    #Plot the neural network loss
    def plotNeuralNet(self,Train,Test):
        #Text Position
        x = 50
        y = (self.loss[-1] + self.loss[0])/2
        #Plot
        plt.plot(self.loss)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Calculated Loss')
        plt.title('Loss curve of training the Neural Network')
        plt.text(x,y,'Training Accuracy: {}'.format(round(Train,2)), fontsize=10)
        plt.text(x,(y + 0.1),'Testing Accuracy: {}'.format(round(Test,2)), fontsize=10)
        plt.show()
