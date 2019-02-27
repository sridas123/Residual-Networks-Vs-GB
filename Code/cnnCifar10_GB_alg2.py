from __future__ import division  # floating point division
import time
import numpy as np
import math
#import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.optimizers import Adam
from keras.regularizers import l2 
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from sklearn.preprocessing import normalize 

"""THIS IS THE NO SAMPLING VERSION"""

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
subset_train= 50000
subset_test=10000
""" 32 examples in a mini-batch, smaller batch size means more updates in one epoch"""

num_classes = 10 #
epochs = 1 # repeat 100 times
no_of_boosting_iter=2

def base_model_reg():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), input_shape=(32,16,16)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='linear'))
    #model.add(Dense(num_classes,activation='softmax'))
    #optimizer = tf.train.RMSPropOptimizer(0.001)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mean_squared_error'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model

def base_initial():

   model = Sequential()
   model.add(Conv2D(32, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), input_shape=x_train.shape[1:]))
   model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.25))
   model.add(Flatten())
   model.add(Dense(num_classes))
   #model.add(Activation('softmax'))
   model.add(Activation('linear'))
   #optimizer = tf.train.RMSPropOptimizer(0.001)
   optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
   """ Train model"""

   model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mean_squared_error'])
   #model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

   return model

"""Calculate the residuals for each training example"""

def calculate_residuals_reg(y_train,y_pred):
    error=[]
    for i in range(0,y_train.shape[0]):
        pred=y_pred[i]
        ytrain=y_train[i]
        residual=[] 
        for j in range(0,pred.shape[0]):
            residual.append(ytrain[j]-pred[j])
        residual=np.asarray(residual)
        error.append(residual)
    error=np.asarray(error)
    return error

def sigmoid(x):
    if (x < -100):
       x=-100
    return(1/(1+math.exp(-x)))

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest)))

def decode(datum):
    return np.argmax(datum)

def splitdataset(x_data, y_data, subset):
    randindices=np.random.randint(0,x_data.shape[0],subset)
    x_subset=x_data[randindices]
    y_subset=y_data[randindices]

    return x_subset,y_subset

def normalization(vec, max, min): 
    temp=[]
    for i in range(0,vec.shape[0]):
	temp.append((vec[i] - min) / (max - min))
    return temp    

def scale(vec):
    temp=[]
    for i in range(0,vec.shape[0]):
	temp.append(vec[i]/np.sum(vec))
    return temp

def calculate_softmax(vec):
    prob_vec=[]
    """Compute softmax values for each sets of scores in x."""
    for i in range(0,vec.shape[0]):
        prob= np.exp(vec[i]) / np.sum(np.exp(vec), axis=0)
        prob_vec.append(prob)
    return prob_vec

def calculate_old_predictions(training_model,xtrain,curr_step):

    #y_pred_reg=np.zeros((xtrain.shape[0],num_classes))
    learner=training_model[curr_step-1]
    y_pred_reg=learner.predict(xtrain, verbose=0)
    return y_pred_reg
      
if __name__ == '__main__':
   
  #sys.stdout = open('analysis.txt', 'w')
  GB_training_model=[]
  int_feature_models=[]

  """get the training and test sets for CIFAR-10"""
  (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

  """Convert and pre-processing"""

  y_train_full = np_utils.to_categorical(y_train_full, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  x_train_full = x_train_full.astype('float32')
  x_test = x_test.astype('float32')
  x_train_full  /= 255
  x_test /= 255
  
  x_train=x_train_full
  """Changed due to sampling"""
  y_train=y_train_full

  """get a subset of x_train and x_test for testing which is less than 50000"""
  """Need to uncomment if you are not doing ensemble wise sampling"""
  #x_train,y_train = splitdataset(x_train_full, y_train_full, subset_train)
  #x_test  ,y_test = splitdataset(x_test, y_test, subset_test)
  
  """Calculate the residuals for a weak model"""
  """Calculate the Initial transformation"""
  cnn_init=base_initial()
  cnn_initial=cnn_init.fit(x_train, y_train, batch_size=batch_size, epochs=epochs ,shuffle=True)
  """Projection of xtrain in 10 dimension"""
  xtrain_projection=cnn_init.predict(x_train,verbose=0)
  cnn_init.summary()
  #print "The shape of x_train is", x_train.shape,x_train.shape[1:]
  
  """Extract the features from the last but 1 layer of the CNN"""
  int_layer_model = Model(input=cnn_init.input, output=cnn_init.get_layer('dropout_1').output)
  int_features = int_layer_model.predict(x_train)
  #print "The intermediate feature  is", int_features[0][:,0:5] 
  
  #print "The projections are",  xtrain_projection[0:2]
  error = calculate_residuals_reg(y_train, xtrain_projection)
  print "xtrain_projection", xtrain_projection[0:5] 
  print "error is", error[0:5]
  #print "The dimension of initial xtrain_projection, error is", xtrain_projection.shape,error.shape

  """Build the CNN model architecture with 1 convolutional and 1 fully connected network for regression"""
  #int_feature_models.append(int_layer_model)
  for steps in range(0,no_of_boosting_iter):
     #print "The int features", int_features.shape[1:]
     cnn_n_reg = base_model_reg()
     """gives out the summary of the models"""
     cnn_n_reg.summary()

     """Fit the Regression model"""
     print "Now fitting"
     cnn = cnn_n_reg.fit(int_features, error, batch_size=batch_size, epochs=epochs ,shuffle=True)
    
     """Predictions from the current regressor"""
     xtrain_projection=cnn_n_reg.predict(int_features, verbose=0)
     cnn_n_reg.summary()

     int_layer_model_reg = Model(input=cnn_n_reg.input, output=cnn_n_reg.get_layer('dropout_'+str(steps+2)).output)
     int_features_reg = int_layer_model_reg.predict(int_features)
     #print "The intermediate feature of reg", int_features_reg[0][:,0:5]

     #print "The predictions from regressor", y_predict_reg_curr[0:5]
     #"""Predictions from previous regressors"""
     #if (steps > 0):
     #   y_predict_prev_models=calculate_old_predictions(GB_training_model,x_train,steps)
     #   """Combine the previous previous and the current prediction"""
     #   #y_predict_reg= np.true_divide(np.add(y_predict_reg_curr,y_predict_prev_models),2)
     #   y_predict_reg= np.add(y_predict_reg_curr,y_predict_prev_models)
     #else:
     #   y_predict_reg= y_predict_reg_curr
     #print "************",xtrain_projection[0],y_predict_reg_curr[0]
     #xtrain_projection=np.add(xtrain_projection, y_predict_reg_curr)
     #print "************AFTER ADDING",xtrain_projection[0]
     
     """Train the new Regression model on intermediate features from the last CNN and the residuals"""
     error=calculate_residuals_reg(y_train,xtrain_projection)
     int_features=int_features_reg
     #print "xtrain_projection_reg", xtrain_projection[0:5]
     #print "error_reg is", error[0:5]
     #print "The dimension of xtrain_projection, error inside Boosting iteration", xtrain_projection.shape,error.shape
     #print "Errors from regressor :: ",error[0:5]
     GB_training_model.append(cnn_n_reg)
     int_feature_models.append(int_layer_model_reg)
     #print len(int_feature_models)
     print "Finished Regression model for step",steps

  """Prediction of data points from the GB model"""
  print "STARTING INFERENCE ON ENSEMBLES"""
  pred_all_model=[]
  
  """While prediction, get the initial feature space to output space mapping from the underlying 1 hidden layer Conv-2D and extract the intermediate features"""
  xtest_projection=cnn_init.predict(x_test,verbose=0) 
  int_features_test=int_layer_model.predict(x_test)
  pred_all_model.append(xtest_projection)
  
  #print "The length of inetrmediate features are**********************************************", len(int_feature_models[:-1]) 
  #for learners in  int_feature_models[:-1]:
  i=0
  for learners in int_feature_models:
      """New Inference for this algorithm which is different than GB (algorithm1)"""
       
      int_feature_test_temp= learners.predict(int_features_test,verbose=0)
      pred_all_model.append(GB_training_model[i].predict(int_features_test))
      int_feature_test=int_feature_test_temp
      i=i+1

  """Preditcion from the last Regressor"""
  #y_pred=GB_training_model[-1].predict(int_feature_test)
  #print "The shape of y_pred is", y_pred.shape 
      #xtest_projection=np.add(y_pred,xtest_projection)
  #pred_entire_model.append(y_pred)
  
  result=np.zeros((x_test.shape[0],num_classes))
  
  """Sums the Regression values for each example and for all classes"""
  for i in range(0,len(pred_all_model)):
      prediction_per_learner=pred_all_model[i]
      result=np.add(result,prediction_per_learner)
  
  pred_entire_model=result
     
  """Calculates the softmax of the regression values"""
  sigmoid_each_elem=[]
  for i in range(0,pred_entire_model.shape[0]):
	softmax_result=calculate_softmax(pred_entire_model[i])
        sigmoid_each_elem.append(softmax_result)

  sigmoid_each_elem=np.asarray(sigmoid_each_elem)
  
  #print "The y_test   ::   ", y_test[0:5]
  #print "softmax from regressors :: ", sigmoid_each_elem[0:5]
 
  y_test_final=sigmoid_each_elem
  
  """Taking the argmax index from the vector"""
  y_test_final_pred=[]
  for i in range(0,y_test_final.shape[0]): 
      y_test_one_ins = y_test_final[i]
      ins_target=np.argmax(y_test_one_ins)
      y_test_final_pred.append(ins_target)
  y_test_final_pred = np.asarray(y_test_final_pred)

  """Decode one hot encodings of y_test"""
  y_test_decoded=[]
  for i in range(y_test.shape[0]):
      y_test_decoded.append(decode(y_test[i]))
  
  y_test_decoded=np.asarray(y_test_decoded)

  """Calculate the accuracy of the network"""
  accuracy=getaccuracy(y_test_decoded,y_test_final_pred) 
  print "*********The accuracy is*****************", accuracy
