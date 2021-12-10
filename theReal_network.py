# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:40:55 2021

@author: agustring
"""
import time
import numpy as np
import urllib.request, json 
import random
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, LSTM, Input,Bidirectional
from tensorflow.keras.optimizers import Adamax, SGD, RMSprop
from keras import metrics

def data_acq():
    dataset = []
    t1 = time.time()
    x=0
    
    def help_the_norman(df,leng):
      a = df
      min_i = 0
      min_j = 0
      for i in range(150-leng,150):
        j = 10
        if df[i,j] < 1e-6:
          min_i = i
        if df[i,j+20] < 1e-6:
          min_j = i
     
      if min_j+1 > len(df)-1:
        min_j = 0
      if min_i+1 > len(df)-1:
        min_i = 0
    
      for i in range(int(min_i+1)):
        for j in range(22):
          a[i,j] = df[min_i+1,j]
          a[i,j+42] = df[min_i+1,j+42]
    
      for i in range(int(min_j+1)):
        for j in range(22,42):
          a[i,j] = df[min_j+1,j]
          a[i,j+42] = df[min_j+1,j+42]
    
      return a
      
    def norman(df):
      a = df
      min_x = df[0][0][0]
      max_x = df[0][0][0]
      min_y = df[0][0][1]
      max_y = df[0][0][1]
      for i in range(len(df)):
        for j in range(len(df[i])):
          if df[i][j][0] < min_x:
            min_x = df[i][j][0]
          if df[i][j][0] > max_x:
            max_x = df[i][j][0]
          if df[i][j][1] < min_y:
            min_y = df[i][j][1]
          if df[i][j][1] > max_y:
            max_y = df[i][j][1]
    
      for i in range(len(df)):
        for j in range(len(df[i])):
          b = (max_x-min_x)
          c = (max_y-min_y)
          if b > 1e-6:
            a[i][j][0] = (df[i][j][0]-min_x)/b
            a[i][j][1] = (df[i][j][1]-min_y)/c
          else:
            a[i][j][0] = 0
            a[i][j][1] = 0
      return a
    
    for user in range(1,11): #1,11
      u = '00'+str(user)
          
      if user>9:
          u = '0'+str(user)
    
      for label in range(1,16): #1,16
          
          l = '00'+str(label)
          
          if label>9:
              l = '0'+str(label)
              
          for sample in range(1,6): #1,6
              t2 = time.time()
              s = '00'+str(sample)
              
              url = 'https://raw.githubusercontent.com/agustring/LSA_classifier/main/preproc%20data2/{}_{}_{}_pre.json?token=ARECJQ4LDX673DAOGWXLV5TBSKZ5W'.format(u, l, s)
              
              with urllib.request.urlopen(url) as url2:
                  df = json.loads(url2.read().decode())
              df = norman(df)
              data_struct = np.array([[0 for i in range(2*42)] for j in range(150)], dtype=float)
              for i in range(len(df)):
                  p=0
                  for j in range(0,42,2):
                      data_struct[i+(150-len(df)),j] = df[i][p][0]
                      data_struct[i+(150-len(df)),j+42] = df[i][p][1]
                      #data_struct[i,j+2] = df[i][p][2]
                      p+=1
              dataset.append(help_the_norman(data_struct,len(df)))
    
              print('Video numero ',x,' en segundos: ',(time.time()-t2))
              x+=1
              print('Todo en minutos: ',(time.time()-t1)/60)
    
                
    dataset = np.array(dataset)

    labels = np.zeros(len(dataset),dtype=int)
    x=-1
    for k in range(0,len(dataset),5):
        x+=1
        if x==15:
            x=0
        labels[k:k+5] = int(x)
        
    labels = np.array(labels)
    
    return dataset, labels

def train_split(dataset, labels):
    tst = .5
    X_train = dataset[:int(len(dataset)*tst)]
    y_train = labels[:int(len(dataset)*tst)]
    x_test = dataset[int(len(dataset)*tst):]
    y_test = labels[int(len(dataset)*tst):]
    
    index = []
    index_lbl = []
    
    for i in range(0,int(len(dataset)),5):
      index_lbl.append(i+2)
      index.append(i)
      index.append(i+1)
      index.append(i+3)
      index.append(i+4)
    
    random.shuffle(index)
    random.shuffle(index_lbl)
    
    X_train = dataset[index]
    y_train = labels[index]
    x_test = dataset[index_lbl]
    y_test = labels[index_lbl]
    return X_train, y_train, x_test, y_test

def pred2label(y_pred):
    index = []
    a = 0
    for i in range(len(y_pred)):
        a = 0
        ind = 0
        for j in range(len(y_pred[i])):
            if y_pred[i][j] > a:
                a = y_pred[i][j]
                ind = j      
        index.append(ind)
    return index

model = Sequential()

model.add(Input((150,84)))
model.add(Bidirectional(LSTM(256,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))

opt = Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.9995, epsilon=1e-07, name="Adamax")
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

dataset, labels = data_acq()
X_train, y_train, x_test, y_test = train_split(dataset, labels)

model.summary()

history = model.fit(X_train, y_train, epochs=500, batch_size=20, shuffle=True, validation_data=(x_test, y_test))
y_pred = model.predict(x_test)

plt.figure(figsize=(12,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()