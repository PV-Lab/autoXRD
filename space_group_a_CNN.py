#-*- coding: utf-8 -*-  
#!usr/bin/env python  

"""
SPACE GROUP a-CNN

filneame: space_group_a_CNN.py version: 1.0
dependencies: 
    autoXRD version 1.0
    autoXRD_vis version 0.2
    
Code to perform classification of XRD patterns for various spcae-group using 
physics-informed data augmentation and all convolutional neural network (a-CNN).
Code to plot class activation maps from a-CNN and global average pooling layer

@authors: Felipe Oviedo and Danny Zekun Ren
MIT Photovoltaics Laboratory / Singapore and MIT Alliance for Research and Tehcnology

All code is under Apache 2.0 license, please cite any use of the code as explained 
in the README.rst file, in the GitHub repository.

"""

################################################################# 
#Libraries and dependencies
#################################################################

# Loads series of functions for preprocessing and data augmentation
from autoXRD import * 
# Loads CAMs visualizations for a-CNN
from autoXRD_vis import * 

import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import KFold

# Neural networks uses Keran with TF background
import keras as K
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

import tensorflow as tf

# Clear Keras and TF session, if run previously
K.backend.clear_session()
tf.reset_default_graph()

# Training Parameters

BATCH_SIZE=128

# Network Parameters
n_input = 1200 # Total angles in XRD pattern
n_classes = 7 # Number of space-group classes
filter_size = 2
kernel_size = 10

################################################################
# Load data and preprocess
################################################################

# Load simulated and anonimized dataset
import os
dirname = os.path.dirname(__file__)

theor = pd.read_csv(os.path.join(dirname, 'Datasets/theor.csv'), index_col=0)
theor = theor.iloc[1:,]
theor_arr=theor.values

# Normalize data for training
ntheor = normdata(theor_arr)

# Load labels for simulated data
label_theo = pd.read_csv(os.path.join(dirname, 'Datasets/label_theo.csv'), header=None, index_col=0)
label_theo = label_theo[1].tolist()

# Load experimental data as dataframe
exp_arr_new = pd.read_csv(os.path.join(dirname, 'Datasets/exp.csv'), index_col=0)
exp_arr_new = exp_arr_new.values

# Load experimental class labels
label_exp= pd.read_csv(os.path.join(dirname, 'Datasets/label_exp.csv'), index_col=0).values
label_exp = label_exp.reshape([len(label_exp),])

# Load class enconding
space_group_enc = pd.read_csv(os.path.join(dirname, 'Datasets/encoding.csv'), index_col=0)
space_group_enc = list(space_group_enc['0'])

# Normalize experimental data
nexp = normdata(exp_arr_new)

# Define spectral range for data augmentation
exp_min = 0
exp_max = 1200 
theor_min = 125

#window size for experimental data extraction
window = 20
theor_max = theor_min+exp_max-exp_min

# Preprocess experimental data
post_exp = normdatasingle(exp_data_processing (nexp, exp_min, exp_max, window))

################################################################
# Perform data augmentation
################################################################

# Specify how many data points we augmented
th_num = 2000

# Augment data, this may take a bit
augd,pard,crop_augd = augdata(ntheor, th_num, label_theo, theor_min, theor_max)    

# Enconde theoretical labels
label_t=np.zeros([len(pard),])
for i in range(len(pard)):
    label_t[i]=space_group_enc.index(pard[i])

# Input the num of experimetal data points       
exp_num =88

# Prepare experimental arrays for training and testing
X_exp = np.transpose(post_exp[:,0:exp_num])
y_exp = label_exp[0:exp_num]

# Prepare simulated arrays for training and testing
X_th = np.transpose(crop_augd )
y_th = label_t

################################################################
# Perform training and cross-validation
################################################################

fold = 5 # Number of k-folds

k_fold = KFold(n_splits=fold, shuffle=True, random_state=3)

# Create arrays to populate metrics
accuracy_exp = np.empty((fold,1))
accuracy_exp_b = np.empty((fold,1)) 
accuracy_exp_r1 = np.empty((fold,1)) 
accuracy_exp_p1 = np.empty((fold,1))
accuracy_exp_r2 = np.empty((fold,1))  
accuracy_exp_p2 = np.empty((fold,1))
f1=np.empty((fold,1))
f1_m=np.empty((fold,1))

# Create auxiliary arrays
accuracy=[]
logs=[]
ground_truth=[]
predictions_ord=[]
trains=[]
tests=[]
trains_combine=[]
trains_y=[]
     
# Run cross validation and define a-CNN each time in loop
for k, (train, test) in enumerate(k_fold.split(X_exp, y_exp)):

        #Save splits for later use
        trains.append(train)
        tests.append(test)
        
        #Data augmentation of experimental traning dataset, we
        # already removed the experimental training dataset
        temp_x = X_exp[train]
        temp_y = y_exp[train]
        exp_train_x,exp_train_y = exp_augdata(temp_x.T,5000,temp_y)
        
        # Combine theoretical and experimenal dataset for training
        train_combine = np.concatenate((X_th,exp_train_x.T))
        trains_combine.append(train_combine)
        
        # Clear weights and networks state
        K.backend.clear_session()

        # Network Parameters
        BATCH_SIZE=128
        n_input = 1200 # MNIST data input (img shape: 28*28)
        n_classes = 7 # MNIST total classes (0-9 digits)
        filter_size = 2
        kernel_size = 10

        enc = OneHotEncoder(sparse=False)
      
        train_dim = train_combine.reshape(train_combine.shape[0],1200,1)
        train_y = np.concatenate((y_th,exp_train_y))
        trains_y.append(train_y)
        train_y_hot = enc.fit_transform(train_y.reshape(-1,1))
        
        # Define network structure
        model = Sequential()

        model.add(K.layers.Conv1D(32, 8,strides=8, padding='same',input_shape=(1200,1), activation='relu'))
        model.add(K.layers.Conv1D(32, 5,strides=5, padding='same', activation='relu'))
        model.add(K.layers.Conv1D(32, 3,strides=3, padding='same', activation='relu'))
        model.add(K.layers.pooling.GlobalAveragePooling1D())
        model.add(K.layers.Dense(n_classes, activation='softmax'))
        
        #Define optimizer        
        optimizer = K.optimizers.Adam()
        
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        
        # Choose early stop
        #early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        
        # Reduce learning rate during optimization
#        reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
#                                      patience=50, min_lr=0.00001)

        # Define test data
        test_x = X_exp[test]
        test_x = test_x.reshape(test_x.shape[0],1200,1)
        test_y = enc.fit_transform(y_exp.reshape(-1,1))[test]
        
        # Fit model
        hist = model.fit(train_dim, train_y_hot, batch_size=BATCH_SIZE, nb_epoch=100,
                         verbose=1, validation_data=(test_x, test_y))
#        hist = model.fit(train_dim, train_y_hot, batch_size=BATCH_SIZE, nb_epoch=100,
#                                     verbose=1, validation_data=(test_x, test_y), callbacks = [early_stop])
#        
        #Compute model predictions
        prediction=model.predict(test_x)
 
       #Go from one-hot to ordinal...
        prediction_ord=[np.argmax(element) for element in prediction]
        predictions_ord.append(prediction_ord)
        
        # Compute accuracy, recall, precision and F1 with macro and micro averaging
        accuracy_exp[k] = metrics.accuracy_score(y_exp[test], prediction_ord) 
        accuracy_exp_r1[k] = metrics.recall_score(y_exp[test], prediction_ord, average='macro') 
        accuracy_exp_r2[k] = metrics.recall_score(y_exp[test], prediction_ord, average='micro')
        accuracy_exp_p1[k] = metrics.precision_score(y_exp[test], prediction_ord, average='macro') 
        accuracy_exp_p2[k] = metrics.precision_score(y_exp[test], prediction_ord, average='micro')
        f1[k]=metrics.f1_score(y_exp[test], prediction_ord, average='micro')
        f1_m[k]=metrics.f1_score(y_exp[test], prediction_ord, average='macro')
        
        #Produce ground_truth, each list element contains array with test elements on first column with respect to X_exp and 
        ground_truth.append(np.concatenate([test.reshape(len(test),1),y_exp[test].reshape(len(test),1)],axis=1))
        
        #Compute loss and accuracy for each k validation
        accuracy.append(model.evaluate(test_x, test_y, verbose=0))
        
        #Save logs
        log = pd.DataFrame(hist.history)
        logs.append(log)
       
        #Save models on current folder with names subscripts 0 to 4
        model.save(os.path.join(dirname, 'keras_model')+str(k)+'.h5')

#
accuracy = np.array(accuracy)        

# Plot final cross validation accuracy
print ('Mean Cross-val accuracy', np.mean(accuracy[:,1]))    

################################################################
# Plotting Class Activation Maps
################################################################

# Compute correctly classified and incorrectly classified cases
corrects, incorrects = find_incorrects(ground_truth,predictions_ord)

# Get dataframe of all incorrects and dataframe of all corrects
corrects_df = pd.concat([r for r in corrects], ignore_index=False, axis=0)
incorrects_df = pd.concat([r for r in incorrects], ignore_index=False, axis=0)

# Get the cam for the trained examples, for each class we average the cam of all trained examples
# Trains refers to the elements in X_exp used for training
# Choose the model in cross validation as output, in this case we plot number 5

cam_outputs=get_cam('keras_model4.h5', trains[4], X_exp)
cam_df=pd.DataFrame(cam_outputs)
cam_df=cam_df.iloc[1:]
cam_df['Label']=y_exp[trains[4]]

# CAM with all augmented training data
rng=range(0,7000)
cam_outputs2=get_cam('keras_model4.h5', rng, train_combine)
cam_df2=pd.DataFrame(cam_outputs2)
cam_df2=cam_df2.iloc[1:]
cam_df2['Label']=train_y

# Now we focus on the incorrectly labelled cam's
incorrects_filtered=incorrects_df[incorrects_df.Model=='keras_model4.h5']

# Get CAM of incorrectly classified cases
cam_inc=get_cam('keras_model4.h5', [int(element) for element in incorrects_filtered.index], X_exp)
cam_inc=pd.DataFrame(cam_inc)
cam_inc=cam_inc.iloc[1:]    

# Redefine main variable
cam_df=cam_df2

#Get the average class maps for a certain class, in this case 6

cam_filtered=cam_df[cam_df.Label==6]
means_6=cam_filtered.mean()
means_6=means_6.iloc[:-1]

#Plot the average class map for class 6
plot_cam(means_6,'Average CAM for Class 6, trained model4.h5')

#Plot correctly classified CAMs and patterns, no need to change this
corrects_filtered=corrects_df[corrects_df.Model=='keras_model4.h5']
cam_cor=get_cam('keras_model4.h5', [int(element) for element in corrects_filtered.index], X_exp)
cam_cor=pd.DataFrame(cam_cor)
cam_cor=cam_cor.iloc[1:]

#Plot a single correctly classified pattern
plot_cam(cam_cor.iloc[-1,:],'CAM for single pattern: true class 6, predicted is 6', X_exp[int(corrects_filtered.index[-1])])

# Note that results may vary due to NN variability during training, make sure you are plotting
# correct and incorrect cases clearly

################################################################