"""
AUTO-XRD

filneame: autoXRD.py version: 1.0
    
Series of functions for XRD spectra pre-processing, normalization
and data augmentation

@authors: Felipe Oviedo and Danny Zekun Ren
MIT Photovoltaics Laboratory / Singapore and MIT Alliance for Research and Tehcnology

All code is under Apache 2.0 license, please cite any use of the code as explained 
in the README.rst file, in the GitHub repository.

"""

################################################################# 
#Libraries and dependencies
################################################################

import pandas as pd
import numpy as np  
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt

################################################################# 
# Functions
#################################################################

# Data normalization from 0 to 1 for double column dataframe, returns single column array
def normdata(data):
    
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1//2])
    for i in range(w1//2):
        ndata[:,i]=(data[:,2*i+1]-min(data[:,2*i+1]))/(max(data[:,2*i+1])-min(data[:,2*i+1]))
    return ndata

# Data normalization from 0 to 1 for single column dataframe
def normdatasingle(data):
    
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1])
    for i in range(w1):
        ndata[:,i]=(data[:,i]-min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
    return ndata

# Data augmendatation for simulated XRD patterns
def augdata(data,num,par1,minn,maxn):
    
    (len1,w1) = np.shape(data)
    augd =np.zeros([len1,num])
    naugd=np.zeros([len1,num])
    newaugd=np.zeros([len1,num])
    crop_augd = np.zeros([maxn-minn,num])
    pard = []
    
    for i in range(num):
        rnd = np.random.randint(0,w1)
        # create the first filter for peak elimination
        dumb= np.repeat(np.random.choice([0,1,1],300),len1//300)
        dumb1= np.append(dumb,np.zeros([len1-len(dumb),]))
        # create the second filter for peak scaling
        dumbrnd= np.repeat(np.random.rand(100,),len1//100)
        dumbrnd1=np.append(dumbrnd,np.zeros([len1-len(dumbrnd),]))
        #peak eleminsation and scaling
        augd[:,i] = np.multiply((data[:,rnd]),dumbrnd1)
        augd[:,i] = np.multiply(augd[:,i],dumb1)
        #nomrlization
        naugd[:,i] = (augd[:,i]-min(augd[:,i]))/(max(augd[:,i])-min(augd[:,i])+1e-9)
        pard.append (par1[2*rnd])
        #adding shift
        cut = np.random.randint(-20*1,20)
        #XRD spectrum shift to left
        if cut>=0:
            newaugd[:,i] = np.append(naugd[cut:,i],np.zeros([cut,]))
        #XRD spectrum shift to right
        else:
            newaugd[:,i] = np.append(naugd[0:len1+cut,i],np.zeros([cut*-1,]))
         
        crop_augd[:,i] = newaugd[minn:maxn,i]
#        
    return newaugd, pard,crop_augd
# Data augmendatation for experimental XRD pattern
def exp_augdata(data,num,label):
    
    (len1,w1) = np.shape(data)
    augd =np.zeros([len1,num])
    naugd=np.zeros([len1,num])
    newaugd=np.zeros([len1,num])
    par=np.zeros([num,])
    
    for i in range(num):
        rnd = np.random.randint(0,w1)

        # create the first filter for peak elimination
        dumb= np.repeat(np.random.choice([0,1,1],300),len1//300)
        dumb1= np.append(dumb,np.zeros([len1-len(dumb),]))
        # create the second filter for peak scaling
        dumbrnd= np.repeat(np.random.rand(200,),len1//200)
        dumbrnd1=np.append(dumbrnd,np.zeros([len1-len(dumbrnd),]))
        #peak eleminsation and scaling
        augd[:,i] = np.multiply((data[:,rnd]),dumbrnd1)
        augd[:,i] = np.multiply(augd[:,i],dumb1)
        #normalization
        naugd[:,i] = (augd[:,i]-min(augd[:,i]))/(max(augd[:,i])-min(augd[:,i])+1e-9)
        par[i,] =label[rnd,] 
        #adding shift
        cut = np.random.randint(-20*1,20)
        #XRD spectrum shift to left
        if cut>=0:
            newaugd[:,i] = np.append(naugd[cut:,i],np.zeros([cut,]))
        #XRD spectrum shift to right
        else:
            newaugd[:,i] = np.append(naugd[0:len1+cut,i],np.zeros([cut*-1,]))

    return newaugd, par

# Extracting experimental data
def exp_data_processing (data,minn,maxn,window):
    
    (len1,w1) = np.shape(data)
    nexp1 =np.zeros([maxn-minn,w1])
    for i in range(w1):
        #savgol_filter to smooth the data
         new1 = savgol_filter(data[minn:maxn,i], 31, 3)
         #peak finding
         zf= find_peaks_cwt(new1, np.arange(10,15), noise_perc=0.01)
         #background substraction
         for j in range(len(zf)-1):
             zf_start= np.maximum(0,zf[j+1]-window//2)
             zf_end = np.minimum(zf[j+1]+window//2,maxn)
             peak = new1[zf_start:zf_end]
             
             ##abritaryly remove 1/4 data
             npeak = np.maximum(0,peak-max(np.partition(peak,window//5 )[0:window//5]))
             nexp1[zf_start:zf_end,i]= npeak     
    return nexp1

##################################################################