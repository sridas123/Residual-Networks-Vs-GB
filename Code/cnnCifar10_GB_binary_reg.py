# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:51:53 2019

@author: nandi
"""

from __future__ import division  # floating point division
import time
import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.optimizers import Adam 
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from sklearn.preprocessing import normalize 
from keras.regularizers import l2


np.set_printoptions(threshold=np.inf)

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

"""Import Tensorflow with multiprocessing"""
import tensorflow as tf
import multiprocessing as mp

""" Loading the CIFAR-10 datasets"""
from keras.datasets import cifar10

""" Declare variables"""

batch_size = 32
subset_train= 2000
subset_test=500
""" 32 examples in a mini-batch, smaller batch size means more updates in one epoch"""

num_classes = 10 #
epochs = 200 # repeat 100 times
no_of_boosting_iter=6

def base_model_reg():

    model = Sequential()
    #model.add(Conv2D(64, (7, 7), strides=(2,2), padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), input_shape=x_train.shape[1:]))    
    model.add(Conv2D(32,(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25)) 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
    #model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Flatten())	
    model.add(Dense(1))
    model.add(Activation('linear'))
    """ Train model"""

    model.compile(loss='mean_squared_error', optimizer=Adam(1e-04),metrics=['mse'])
    return model

"""Calculate the residuals (I-P) for each training example"""
def calculate_residuals_cls(y_train,y_pred):   
    e1=[]
    for i in range(0,y_train.shape[0]):
        pred=y_pred[i]
        residual=[]
        for j in range(0,pred.shape[0]): 
            residual.append(y_train[i][j]-pred[j])
        residual=np.asarray(residual)
        e1.append(residual)
    e1=np.asarray(e1)
    return e1

def calculate_residuals_reg(y_train,e1_predicted,y_predicted1):
    """e1=calculate_residuals_reg(y_train_full,e1_predicted,y_predicted1)"""
    e1=[]
    for i in range(0,y_train.shape[0]):
        pred=e1_predicted[i]
        prev=y_predicted1[i]
        ytrain=y_train[i]
        residual=[]
        for j in range(0,pred.shape[0]):
            residual.append(ytrain[j]-((pred[j]+prev[j])/2))
        residual=np.asarray(residual)
        e1.append(residual)
    e1=np.asarray(e1)
    return e1

def sigmoid(x):
    if (x < -100):
       x=-100
    return(1/(1+math.exp(-x)))

def inverse_sigmoid(x):
    return(math.log(x/(1-x)))
	
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest)))

def decode(datum):
    return np.argmax(datum)

def splitdataset(x_data, y_data):
    pos_count=5000
    neg_count=5000
    index_positives=np.where(y_data > 0)[0]
    pos=np.random.choice(index_positives, pos_count)
    index_negatives= np.where(y_data==0)[0]
    negs=np.random.choice(index_negatives, neg_count)
    tot=np.concatenate((pos,negs))
    x_subset=x_data[tot]
    y_subset=y_data[tot]

    return x_subset,y_subset
	
def splitdataset_reg(x_data, y_data, e1):
    pos_count=5000
    neg_count=5000
    index_positives=np.where(y_data > 0)[0]
    pos=np.random.choice(index_positives, pos_count)
    index_negatives= np.where(y_data==0)[0]
    negs=np.random.choice(index_negatives,neg_count)
    tot=np.concatenate((pos,negs))
    x_subset=x_data[tot]
    y_subset=e1[tot]

    return x_subset,y_subset

	
def change_to_binary(data_label,target):
    for i in range(0,data_label.shape[0]):
        if data_label[i]!= target:
           data_label[i]=0
        else:
           data_label[i]=1 
    return data_label    
    
if __name__ == '__main__':
  
  Total_training_model=[]
  """get the training and test sets for CIFAR-10"""
  for i in range(0,num_classes):
      """start learning for each class in target"""
      print "target class being learnt", i
      (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
      y_test_Final=y_test
      y_test_Final = np_utils.to_categorical(y_test_Final, num_classes)
      target_class=i
      
      """Convert and pre-processing"""
      y_train_full=change_to_binary(y_train_full,target_class)
      y_test=change_to_binary(y_test,target_class)
      x_train_full = x_train_full.astype('float32')
      x_test = x_test.astype('float32')
      x_train_full  /= 255
      x_test /= 255      
      
      """Fit the Classification  model"""
      y_predicted1=np.full((y_train_full.shape[0], 1), 0.50)
      GB_training_model=[]
      """Compute residuals from classifier Calculate error residuals. Actual target value, minus predicted target value """
      e1 = calculate_residuals_cls(y_train_full,y_predicted1)
      prev=y_predicted1
      for steps in range(0,no_of_boosting_iter):
          """Perform sampling for next weak regressor"""
          x_train,y_train = splitdataset_reg(x_train_full,y_train_full, e1)
          cnn_n_reg = base_model_reg()
          """gives out the summary of the models"""
          cnn_n_reg.summary()        
          """Fit the Regression model on error residuals as target variable with same input variables """
          cnn = cnn_n_reg.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.0, validation_data=None,shuffle=True)
          """call it e1_predicted"""
          e1_predicted=cnn_n_reg.predict(x_train_full, verbose=0)
          """Add the predicted residuals to the previous predictions ([y_predicted2 = y_predicted1 + e1_predicted]) and  [e2 = y - y_predicted2]"""
          e1=calculate_residuals_reg(y_train_full,e1_predicted,prev)
          prev=e1_predicted
          GB_training_model.append(cnn_n_reg)
          print "Finished Regression model for step",steps
      Total_training_model.append(GB_training_model)
  Total_training_model = np.asarray(Total_training_model)
  #print "Shape of all models ::",Total_training_model.shape
  print "STARTING INFERENCE ON ENSEMBLES"""
  inference_all_Classes=[]
  #y_test_final=[]
  final=[]
  for i in  range(0, Total_training_model.shape[0]):
      pred_each_model=[]
      for each_learner in Total_training_model[i]:
          y_pred= each_learner.predict(x_test,verbose=0)
          pred_each_model.append(y_pred)
      result=np.zeros((x_test.shape[0],1))
      for i in range(0,len(pred_each_model)):
          prediction_per_learner=pred_each_model[i]
          result=np.add(result,prediction_per_learner)
      sigmoid_each_elem=[]
      for i in range(0,result.shape[0]):
          final_pred_test=[]
          for j in range(0,result.shape[1]):
              final_pred_test.append(sigmoid(result[i][j]))			  
              sigmoid_each_elem.append(final_pred_test)
      #sigmoid_each_elem=np.asarray(sigmoid_each_elem)
      #y_test_final=np.add(sigmoid_each_elem,pred_each_model[0])
      #print "y_test_final",y_test_final[0:100],len(y_test_final)
      #y_test_final=y_test_final/2
      #y_test_final=y_test_final.tolist()
      final.append(sigmoid_each_elem)
  #print "y_test_final",final,len(final),len(final[0])
  #dic.setdefault(key,[]).append(value)
  d={}
  for cl in range(0,len(final)):
      for ex in range(0,len(final[0])):
          #print "final[cl][ex] :: ",cl, ex,final[cl][ex]
          if ex not in d:
              d.setdefault(ex,[]).append(final[cl][ex][0])
          else:
              d[ex].append(final[cl][ex][0])
  #print "dict :: ",d
  predictions=[]
  for k in d:
      predictions.append(d[k].index(max(d[k])))
  y_test_decoded=[]
  for i in range(y_test_Final.shape[0]):
      y_test_decoded.append(decode(y_test_Final[i]))
  #print "predictions",predictions
  accuracy=getaccuracy(y_test_decoded,predictions)
  print " y_test_decoded :: ", y_test_decoded[0:200]
  print " predictions :: ", predictions[0:200]
  print "*********The accuracy is*****************", accuracy
