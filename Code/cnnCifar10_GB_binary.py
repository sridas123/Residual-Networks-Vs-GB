# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:51:53 2019

@author: nandi
"""

from __future__ import division  # floating point division
import time
import numpy as np
import math
from keras.models import Sequential
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


np.set_printoptions(threshold=np.inf)

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

"""Import Tensorflow with multiprocessing"""
import tensorflow as tf
import multiprocessing as mp

""" Loading the CIFAR-10 datasets"""
from keras.datasets import cifar10

""" Declare variables"""

batch_size = 128
subset_train= 2000
subset_test=500
""" 32 examples in a mini-batch, smaller batch size means more updates in one epoch"""

num_classes = 10 #
epochs = 1000 # repeat 100 times
no_of_boosting_iter=9

def base_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    """ Train model"""
    model.compile(optimizer=Adam(1e-04),loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def base_model_reg():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())	
    model.add(Dense(1))
    model.add(Activation('linear'))

    #sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = tf.train.RMSPropOptimizer(0.001)
    """ Train model"""

    model.compile(loss='mean_squared_error', optimizer=Adam(1e-04),metrics=['mse'])
    #model.compile(optimizer=Adam(1e-04),loss='mean_squared_error',metrics=['accuracy'])
    return model

"""Calculate the residuals (I-P) for each training example"""
def calculate_residuals_cls(y_train,y_pred):   
    e1=[]
    #print "The shape of y_train and y_pred is", y_train.shape, y_pred.shape
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
            residual.append(ytrain[j]-(sigmoid((pred[j]+prev[j])/2)))
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
    pos_count=4000
    neg_count=12000
    index_positives=np.where(y_data > 0)[0]
    pos=np.random.choice(index_positives, pos_count)
    #index_positives= np.where(y_data>0)
    index_negatives= np.where(y_data==0)[0]
    #print index_positives.shape, index_negatives.shape
    negs=np.random.choice(index_negatives, neg_count)
    tot=np.concatenate((pos,negs))
    x_subset=x_data[tot]
    y_subset=y_data[tot]

    return x_subset,y_subset
	
def splitdataset_reg(x_data, y_data, e1):
    pos_count=4000
    neg_count=12000
    index_positives=np.where(y_data > 0)[0]
    pos=np.random.choice(index_positives, pos_count)
    #index_positives= np.where(y_data>0)
    index_negatives= np.where(y_data==0)[0]
    #print index_positives.shape, index_negatives.shape
    negs=np.random.choice(index_negatives,neg_count)
    tot=np.concatenate((pos,negs))
    x_subset=x_data[tot]
    y_subset=e1[tot]

    return x_subset,y_subset

def normalization(vec, max, min): 
    temp=[]
    for i in range(0,vec.shape[0]):
        p.append((vec[i] - min) / (max - min))
    return temp    

def scale(vec):
    temp=[]
    for i in range(0,vec.shape[0]):
        p.append(vec[i]/np.sum(vec))
    return temp
	
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
      
      """get a subset of train based on undersampling majority class"""
      x_train,y_train = splitdataset(x_train_full, y_train_full)
      """Build the CNN model architecture with 1 convolutional and 1 fully connected network""" 
      cnn_n = base_model()
      """gives out the summary of the models"""
      cnn_n.summary()
      #print "ytrain , ytest :: ", y_train[0:20],y_test[0:20]
    
      """Fit the Classification  model"""
      cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.0, validation_data=None,shuffle=True)
      y_predicted1=cnn_n.predict(x_train_full, verbose=0)
      GB_training_model=[]
      """Compute residuals from classifier Calculate error residuals. Actual target value, minus predicted target value """
      e1 = calculate_residuals_cls(y_train_full,y_predicted1)
      GB_training_model.append(cnn_n)
      y_prev = y_predicted1
      for steps in range(1,no_of_boosting_iter):
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
          e1=calculate_residuals_reg(y_train_full,e1_predicted,y_prev)
          y_prev=e1_predicted
          #print e1[:200]
          GB_training_model.append(cnn_n_reg)
          print "Finished Regression model for step",steps
      Total_training_model.append(GB_training_model)
  Total_training_model = np.asarray(Total_training_model)
  #print "Shape of all models ::",Total_training_model.shape
  print "STARTING INFERENCE ON ENSEMBLES"""
  inference_all_Classes=[]
  y_test_final=[]
  final=[]
  for i in  range(0, Total_training_model.shape[0]):
      pred_each_model=[]
      for each_learner in Total_training_model[i]:
          y_pred= each_learner.predict(x_test,verbose=0)
          pred_each_model.append(y_pred)
      result=np.zeros((x_test.shape[0],1))
      for i in range(1,len(pred_each_model)):
          prediction_per_learner=pred_each_model[i]
          result=np.add(result,prediction_per_learner)
	  #print result.shape
      sigmoid_each_elem=[]
      for i in range(0,result.shape[0]):
          final_pred_test=[]
          for j in range(0,result.shape[1]):
              final_pred_test.append(sigmoid(result[i][j]))
			  
              sigmoid_each_elem.append(final_pred_test)
      sigmoid_each_elem=np.asarray(sigmoid_each_elem)
      #print "sigmoid_each_elem :: ",sigmoid_each_elem.shape
      #print "pred_each_model :: ",pred_each_model.shape
      y_test_final=np.add(sigmoid_each_elem,pred_each_model[0])
      #print "y_test_final",y_test_final[0:100],len(y_test_final)
      y_test_final=y_test_final/2
      y_test_final=y_test_final.tolist()
      final.append(y_test_final)
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
