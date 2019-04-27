# -*- coding: utf-8 -*-
"""
AUTO-XRD VISUALIZATION

filneame: autoXRD_vis.py version: 0.2
    
Series of functions for extracting correctly and incorrectly classified patterns
of XRD cross-validation, 

@authors: Felipe Oviedo and Danny Zekun Ren
MIT Photovoltaics Laboratory / Singapore and MIT Alliance for Research and Tehcnology

All code is under Apache 2.0 license, please cite any use of the code as explained 
in the README.rst file, in the GitHub repository.

"""

#%%
#Function, plot the cam and spectra of certain set of samples
    
from keras.models import load_model
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd

# Function to extract CAM of a certain model, subset of test elements, and spectra array with all experimental data (non-augmented)
def get_cam(model_name, elements, spectra):
    
    model = load_model(model_name)        
    gap_weights = model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=model.input, 
                    outputs=(model.layers[-3].output, model.layers[-1].output))
    
    features, results = cam_model.predict(spectra[elements].reshape(spectra[elements].shape[0],1200,1))
    cam_outputs=np.zeros([10,1])
    
# Extract Class Activation MAPS
    for idx,element in enumerate(elements):  
        
        # get the feature map of the test image
        features_for_one_img = features[idx, :, :]
            
        # map the feature map to the original size
        cam_features = features_for_one_img
            
        # get the predicted label with the maximum probability
        pred = np.argmax(results[idx])
        
        # get the weights of class activation map
        cam_weights = gap_weights[:, pred]
    
        # create the class activation map
        cam_output = np.dot(cam_features, cam_weights)
        cam_output = abs(cam_output)
        cam_outputs=np.append(cam_outputs, cam_output.reshape(cam_output.shape[0],1), axis=1)
        
    return cam_outputs.T

# Plot CLass Activation Map of both an individual CAM and an average CAM
def plot_cam(cam_output, title, *args):
    
    #*arg is here a vector with the actual spectra
    x = np.linspace(10,69.96,cam_output.shape[0])
    y = cam_output
    y2=[]
    for arg in args:
        y2.append(arg)
    if len(y2) != 0:
        y2=np.array(y2).reshape(1200,)
    
    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
    
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent, interpolation='gaussian')
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    
    if len(y2) != 0:
        ax2.plot(np.linspace(10,69.96,1200),y2)
        ax2.set_xlabel(r"$2 \theta$ (Degrees)")
        ax2.set_title("Pattern")
        
    
    fig.suptitle(title, fontsize=16)
    
        
#Find corrects and incorrects in all models ran, and compare them to ground-truth
def find_incorrects(ground_truth,predictions_ord):
    
    corrects=[]
    incorrects=[]
    k=0
    #Create vector of predictions
    for truth, predictionn in zip(ground_truth,predictions_ord):
        
        #Create temp array of predictions
        temp=np.array(predictionn).reshape(len(predictionn),1)
        
        #Join predictions and ground truth and convert to dataframe
        comparision_array=np.concatenate([truth,temp],axis=1)
        comparision_df=pd.DataFrame(data=comparision_array[:,1:], index=comparision_array[:,0], columns=['Truth','Prediction'])
        comparision_df['Model']='keras_model'+str(k)+'.h5'
        
        #Find incorrects and save dataframe
        incorrect_df=comparision_df[comparision_df.Truth != comparision_df.Prediction]
        correct_df=comparision_df[comparision_df.Truth == comparision_df.Prediction]
        #Save incorrects and comparision
        incorrects.append(incorrect_df)
        corrects.append(correct_df)  
        k += 1
    
    #Return list of dataframe with comparision between resuls, and list of dataframes of incorrects
    return corrects, incorrects
