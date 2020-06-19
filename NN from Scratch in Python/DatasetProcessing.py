# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 05:29:12 2020

@author: kartik

This file pre-processes datasets for the neural network. 
Few datasets are included as functions within this file. Their description is 
mentioned with respective function's section. All functions give features and 
target arrays as output.

"""
import pandas as pd

#Dataset 1: Identify the accent

def Accent():
    '''
     Data set from UCI Machine Learning Repository featuring single English 
     words read by speakers from six different countries for accent detection 
     and recognition.
     
     Attributes = 12 features extracted from audio clips
     Target Variables: language = {ES, FR, GE, IT, UK, US} 
     The six possible accents considered
     
     Total Instances = 329
     

    '''
    Accent_data = pd.read_csv('Accent_Dataset.csv')
    OneHot = pd.get_dummies(Accent_data['language'], prefix = 'language')
    Accent_data = pd.concat([Accent_data, OneHot], axis = 1)
    Accent_data = Accent_data.drop(columns = ['language']) #Target
    Accent_data = Accent_data.sample(frac = 1)

    #Separate Target and Input
    data_rows = 329
    X = Accent_data.iloc[0:data_rows,0:12]
    target = Accent_data.iloc[0:data_rows, 12:]
    target = target.values.reshape(X.shape[0],6)
    
    return X, target

#Dataset 2: Heart Condition Dataset

def HeartCondition(n):
    '''
    This dataset contains 13 features and is used to predict wether the subject has
    heart condition or not. [Binary Classification]
    
    Total Instances = 270
    n = number of Instances required for the NN (user specified)
    '''

    Heart_data = pd.read_csv('Sample_Dataset_Heart.csv')
    Heart_data['HeartDisease'] = Heart_data['HeartDisease'].replace(1,0)
    Heart_data['HeartDisease'] = Heart_data['HeartDisease'].replace(2,1) 
    Heart_data = Heart_data.sample(frac = 1)

    #Separate Target and Input
    data_rows = n
    X = Heart_data.iloc[0:data_rows,0:13]
    target = Heart_data.iloc[0:data_rows, -1]
    target = target.values.reshape(X.shape[0],1)
    
    return X, target

#Dataset 3: My own Spotify Playlist Dataset

def Spotifydata(n):
    '''
    This dataset contains features for 400 songs extracted using Spotify API. 
    The dataset has 6 audio features and it classifies songs on the basis of their vibe (personally classified)
    [Multi-CLass Classification]
    
    Total Instances: 400
    n = number of Instances required for the NN (user specified)
    
    '''
    Spotify_data = pd.read_csv('Spotify_Train.csv')
    OneHot = pd.get_dummies(Spotify_data['Mood'], prefix = 'Mood')
    Spotify_data = pd.concat([Spotify_data, OneHot], axis = 1)
    Spotify_data = Spotify_data.drop(columns = ['Name']) #Non-Essential
    Spotify_data = Spotify_data.drop(columns = ['Song_ID']) #Non-Essential
    Spotify_data = Spotify_data.drop(columns = ['Mood']) #Target
    Spotify_data = Spotify_data.sample(frac = 1)

    #Separate Target and Input
    data_rows = n
    X = Spotify_data.iloc[0:data_rows,0:6]
    target = Spotify_data.iloc[0:data_rows,6:11]
    target = target.values.reshape(X.shape[0],5)

    return X, target