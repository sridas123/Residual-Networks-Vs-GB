from __future__ import division  # floating point division
import time
import numpy as np
import math
#import sys
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

"""THIS IS THE SAMPLING VERSION OF THE CODE"""
""" Loading the CIFAR-10 datasets"""
from keras.datasets import cifar10

""" Declare variables"""

batch_size = 128
subset_train= 50000
subset_test=10000
""" 32 examples in a mini-batch, smaller batch size means more updates in one epoch"""

num_classes = 10 #
epochs = 50 # repeat 100 times
no_of_boosting_iter=1

def base_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    #sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.train.RMSPropOptimizer(0.001)

    """ Train model"""

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def base_model_reg():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    #model.add(Conv2D(32, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())	
    model.add(Dense(num_classes))
    model.add(Activation('linear'))

    #sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    """ Train model"""

    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
    #model.compile(optimizer=Adam(1e-04),loss='mean_squared_error',metrics=['accuracy'])
    return model

"""Calculate the residuals (I-P) for each training example"""
def calculate_residuals_cls(y_train):
    prob=(1/num_classes)   
    error=[]
    #print "The shape of y_train and y_pred is", y_train.shape, y_pred.shape
    for i in range(0,y_train.shape[0]):
        #pred=y_pred[i]
        residual=[]
        for j in range(0,num_classes): 
            residual.append(y_train[i][j]-prob)
        residual=np.asarray(residual)
        error.append(residual)
    error=np.asarray(error)
    return error

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
  """get the training and test sets for CIFAR-10"""
  (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

  """Convert and pre-processing"""

  y_train_full = np_utils.to_categorical(y_train_full, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  x_train_full = x_train_full.astype('float32')
  x_test = x_test.astype('float32')
  x_train_full  /= 255
  x_test /= 255
  
  #print "*********The shape of x_train,y_train****************",x_train.shape,y_train.shape
  #x_train=x_train_full
  """Changed due to sampling"""
  y_train=y_train_full

  """get a subset of x_train and x_test for testing which is less than 50000"""
  """Need to uncomment if you are not doing ensemble wise sampling"""
  #x_train,y_train = splitdataset(x_train_full, y_train_full, subset_train)
  #x_test  ,y_test = splitdataset(x_test, y_test, subset_test)
  #print "*********The shape of x_train,y_train****************",x_train.shape,y_train.shape,x_test.shape,y_test.shape
  
  """Calculate the residuals for a weak model"""
  print "y_train is", y_train[0:5] 
  error = calculate_residuals_cls(y_train)
  print "The residuals to start for the task are  ::",  error[0:5]

  """Fit a regression model on the multi dimensional continuous values using KERAS"""
  """Build the CNN model architecture with 1 convolutional and 1 fully connected network for regression"""
  for steps in range(0,no_of_boosting_iter):
     """Sampling step for each ensemble"""
     x_train,error = splitdataset(x_train_full, error, subset_train)
     cnn_n_reg = base_model_reg()
     """gives out the summary of the models"""
     #cnn_n_reg.summary()

     """Fit the Regression model"""
     cnn = cnn_n_reg.fit(x_train, error, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test) ,shuffle=True)
    
     """Predictions from the current regressor"""
     #y_predict_reg_curr=cnn_n_reg.predict(x_train, verbose=0)
     """Sampling loop"""
     y_predict_reg_curr=cnn_n_reg.predict(x_train_full, verbose=0)
     print "The predictions from regressor", y_predict_reg_curr[0:5]
     """Predictions from previous regressors"""
     if (steps > 0):
        #y_predict_prev_models=calculate_old_predictions(GB_training_model,x_train,steps)
        """Sampling loop"""
        y_predict_prev_models=calculate_old_predictions(GB_training_model,x_train_full,steps)
        """Cobine the previous previous and the current prediction"""
        y_predict_reg= np.true_divide(np.add(y_predict_reg_curr,y_predict_prev_models),2)
        #y_predict_reg= np.add(y_predict_reg_curr,y_predict_prev_models)
     else:
        y_predict_reg= y_predict_reg_curr
     
     error=calculate_residuals_reg(y_train,y_predict_reg)
     print "Errors from regressor :: ",error[0:5]
     GB_training_model.append(cnn_n_reg)
     print "Finished Regression model for step",steps

  """Prediction of data points from the GB model"""
  print "STARTING INFERENCE ON ENSEMBLES"""
  pred_each_model=[]
  
  """For testing all x_test has been changed to x_train"""
  for learners in  GB_training_model:
     # y_pred= learners.predict(x_test, batch_size=None, verbose=0, steps=None)
       y_pred= learners.predict(x_test,verbose=0)
       pred_each_model.append(y_pred)
  
  result=np.zeros((x_test.shape[0],num_classes))

  """Sums the Regression values for each example and for all classes"""
  for i in range(0,len(pred_each_model)):
      prediction_per_learner=pred_each_model[i]
      result=np.add(result,prediction_per_learner)
      
  #print "The result is", result[:20]

  """Calculates the softmax of the regression values"""
  sigmoid_each_elem=[]
  for i in range(0,result.shape[0]):
	softmax_result=calculate_softmax(result[i])
        sigmoid_each_elem.append(softmax_result)

  sigmoid_each_elem=np.asarray(sigmoid_each_elem)
  
  print "The y_test   ::   ", y_test[0:5]
  print "softmax from regressors :: ", sigmoid_each_elem[0:5]
  #print "prediction of each model from classifier is  ::",  pred_each_model[0][0:5] 
  
  #y_test_final=[]
 
  #"""Finally add the probabilities of the classifier and sigmoid of the regressors"""
  #for k in range(0,pred_each_model[0].shape[0]):
  #    vec1_cls=pred_each_model[0][k]
  #    vec1_reg=sigmoid_each_elem[k]
  #    vec_sum=np.add(vec1_cls,vec1_reg)
  #    vec_sum=np.divide(vec_sum,2)
  #    y_test_final.append(vec_sum)
  #y_test_final=np.asarray(y_test_final)
  y_test_final=sigmoid_each_elem
  
  #print "Result of combining softmax and classification::", y_test_final[0:5]
  """Taking the argmax index from the vector"""
  y_test_final_pred=[]
  for i in range(0,y_test_final.shape[0]): 
      y_test_one_ins = y_test_final[i]
      ins_target=np.argmax(y_test_one_ins)
      y_test_final_pred.append(ins_target)
  y_test_final_pred = np.asarray(y_test_final_pred)
 
  #print "The y_test   ::   ", y_test[0:5]
  """Decode one hot encodings of y_test"""
  y_test_decoded=[]
  for i in range(y_test.shape[0]):
      y_test_decoded.append(decode(y_test[i]))
  
  y_test_decoded=np.asarray(y_test_decoded)
  #print "y_test_decoded*****",y_test_decoded[:10]
  """Calculate the accuracy of the network"""
  accuracy=getaccuracy(y_test_decoded,y_test_final_pred) 
  print "*********The accuracy is*****************", accuracy
