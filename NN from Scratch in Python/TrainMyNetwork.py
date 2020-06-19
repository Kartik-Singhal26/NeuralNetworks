# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:50:01 2020

@author: kartik

"""
from NeuralNet import NeuralNetwork_2HiddenLayers, NeuralNetwork_3HiddenLayers
from DatasetProcessing import Accent, Spotifydata, HeartCondition 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, target = Spotifydata(250)
#Split Data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, target, test_size = 0.2, random_state = 15)

#Scale Data to standard
Standard_data = StandardScaler()
Standard_data.fit(Xtrain)
Xtrain = Standard_data.transform(Xtrain)
Xtest = Standard_data.transform(Xtest)

#Define your parameters
layers = [256,64,32]
learning_rate = 0.0124
epoches = 5000

#Initialize the neuralnetwork
NeuralNet = NeuralNetwork_3HiddenLayers(Xtrain, Ytrain, layers, learning_rate, epoches)
    
#Train the neuralnetwork
NeuralNet.Training(Xtrain, Ytrain, learning_rate, epoches)

#Model Prediction
Test_Prediction = NeuralNet.Predict(Xtest)
Train_Predicition = NeuralNet.Predict(Xtrain)

#Model Accuracy and Plot Log(Loss) Curve
Test_Accuracy = NeuralNet.accuracy(Xtest, Ytest)
print('The model testing accuracy is: ', round(Test_Accuracy,2))
Train_Accuracy = NeuralNet.accuracy(Xtrain, Ytrain)
print('The model training accuracy is: ', round(Train_Accuracy,2))

NeuralNet.plotNeuralNet(Train_Accuracy,Test_Accuracy)
