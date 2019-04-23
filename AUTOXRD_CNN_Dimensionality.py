# -*- coding: utf-8 -*-
"""
AUTO-XRD V 0.9

Code to perform classification of XRD patterns for various dimensionalities using 
physics-informed data augmentation and all convolutional neural network (a-CNN).

@authors: Felipe Oviedo and Danny Zekun Ren

Code is under MIT license, please cite any use of the code as explained 
in README, in the GitHub repository.

"""


#Libraries
import warnings
warnings.filterwarnings("ignore")
import time  
from sklearn import metrics
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
import keras as K
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping
import tensorflow as tf

#Clear Keras session for repeated running
K.backend.clear_session()

# a-CNN Hyperparemeters

BATCH_SIZE=128

# Network Parameters
n_input = 1200 
n_classes = 3 

enc = OneHotEncoder(sparse=False)
##########FUNCTIONS#######################

#Gets .ASC files from directory
def spectra_list(path,excluded):
    file_pth= [os.path.join(d, x) for d, dirs, files in os.walk(path) for x in files if x.endswith(".ASC") and excluded not in x]
    return file_pth
#Gets .XY files from directory
def spectra_list2(path):
    file_pth= [os.path.join(d, x) for d, dirs, files in os.walk(path) for x in files if x.endswith(".xy")]
    return file_pth
#Groups all curves within a symmetry group into as single dataframe
def group(spectra,k):
    
    groups=[]
    for indx,vals in enumerate(spectra[k]):
        groups.append(pd.read_csv(spectra[k][indx], delim_whitespace=True, header=None))
        df=pd.concat(groups, axis=1)
    return df

#Data normalization from 0 to 1 for double column dataframe, returns single column array
def normdata(data):
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1//2])
    for i in range(w1//2):
        ndata[:,i]=(data[:,2*i+1]-min(data[:,2*i+1]))/(max(data[:,2*i+1])-min(data[:,2*i+1]))
    return ndata
#Data normalization from 0 to 1 for single column dataframe
def normdatasingle(data):
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1])
    for i in range(w1):
        ndata[:,i]=(data[:,i]-min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
    return ndata

#Data augmendatation for simulated XRD patterns
def augdata(data,num,dframe,minn,maxn):
    (len1,w1) = np.shape(data)
    augd =np.zeros([len1,num])
    naugd=np.zeros([len1,num])
    newaugd=np.zeros([len1,num])
    crop_augd = np.zeros([maxn-minn,num])
    par1 = list(dframe.columns.values)
    pard = []
    for i in range(num):
        rnd = np.random.randint(0,w1)
        # create the first filter for peak elimination
        dumb= np.repeat(np.random.choice([0,1,1],300),len1//300)
        dumb1= np.append(dumb,np.zeros([len1-len(dumb),]))
        # create the second filter for peak scaling
        dumbrnd= np.repeat(np.random.rand(100,),len1//100)
        dumbrnd1=np.append(dumbrnd,np.zeros([len1-len(dumbrnd),]))
        # peak elemination and scaling
        augd[:,i] = np.multiply((data[:,rnd]),dumbrnd1)
        augd[:,i] = np.multiply(augd[:,i],dumb1)
        #normalization
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
      
    return newaugd, pard,crop_augd

#data augmendatation for experimental XRD patterns
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
        #peak elimination and scaling
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

#extracting experimental data
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

#####################BEGINNING OF SCRIPT
    
#os.chdir(r"/Users/danny/Dropbox/work/Campaign/XRD dimen/Theoretical")
os.chdir(r"C:\Users\fovie\Dropbox (MIT)\XRD dimen\Theoretical")
#Produce lists with sample names
excluded="test.ASC"
directory=[]
for dir,dirs, files in os.walk("."):
    
    directory.extend(dirs)

spectra_th=[]
for k in directory:
    spectra_th.append(spectra_list(k,excluded))

#Create dictionary of dataframes for each symmetry group    
x = directory[:3]
grp = {}
for indx,vals in enumerate(x):
    grp['%s' %vals] = group(spectra_th,indx)

theor=pd.concat(grp,axis=1)
theor_arr=theor.values

#%Normalize data
ntheor = normdata (theor_arr)

#os.chdir(r"/Users/danny/Dropbox/work/Campaign/XRD dimen/Experimental")
os.chdir(r"C:\Users\fovie\Dropbox (MIT)\XRD dimen\Experimental")
#Produce lists with sample names
directory2=[]
for dir,dirs, files in os.walk("."):
    directory2.extend(dirs)

spectra_exp=[]
for k in directory2[:3]:
    spectra_exp.append(spectra_list2(k))


#Create dictionary of dataframes for each symmetry group    
x = directory2[:3]
grp = {}
for indx,vals in enumerate(x):
    grp['%s' %vals] = group(spectra_exp,indx)
    
#This is the final dataframe and array for the experimental data,
#each spectra is a pair of columns one for angle the other for intensity (not normalized)
#some columns have more data points that others, those who have less have NaN in the empty spaces
exp=pd.concat(grp,axis=1)


#%Preprocessing experimental

#Take out NaN
exp= exp.fillna(method = 'ffill')


#Interpolate for spectra of different angle
exp_arr=exp.values

ang=np.arange(10,70,0.04)

exp_arr_new = np.zeros ([len(ang),len(exp.T)]) 



for i in range(len(exp.T)//2):
    
    
    exp_arr_new [:,2*i] = ang
    
    exp_arr_new [:,2*i+1] = np.interp(ang, exp_arr[:,2*i], exp_arr[:,2*i+1])

#We didn't simulate the peak at 5.00 degrees, so start from 5.04
exp_arr_new=exp_arr_new[1:,:]


#%
#normalization
nexp = normdata (exp_arr_new)
#define the range for spectrum
exp_min = 0
exp_max = 1200 
theor_min = 125
#window size for experimental data extraction
window =20
theor_max = theor_min+exp_max-exp_min
#experimetal data input
post_exp= normdatasingle(exp_data_processing (nexp,exp_min,exp_max,window))

#%
#let's start to do the data augmentation.
#specify how many data points we augmented
acc={}
err= {}
for theor_iter in range(1):
    
    dd = [800]
    theor_num = dd[theor_iter]
    
    augd,pard,crop_augd = augdata(ntheor,theor_num,theor,theor_min,theor_max)    
    
    
    #convert theorectical label from space group to numbers
    label_t=np.zeros([len(pard),])
    for i in range(len(pard)):
        
        temp = pard[i]
        label_t[i]=directory.index(temp[0])
    
     #convert experimental label from space group to numbers       
    
    par_exp = list(exp.columns.values)
    
    label_exp=np.zeros([len(par_exp)//2,])
    
    for i in range(len(par_exp)//2):
        
        temp = par_exp[2*i]
        label_exp[i]=directory2.index(temp[0])
        
    #input the experimetal data points       
    exp_num =124 
    
    #train and test split for the experimental data
    X_exp = np.transpose(post_exp[:,0:exp_num])
    y_exp = label_exp[0:exp_num]
    
    y_exp_hot = enc.fit_transform(y_exp.reshape(-1,1))
    #X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp
    #        ,y_exp , test_size=0.33,random_state=1)
    
    
    
    #train and test split for the theorectical data
    X_th = np.transpose(crop_augd )
    y_th = label_t
    #y_th_hot = enc.fit_transform(y_th.reshape(-1,1))
    #X_train_th, X_test_th, y_train_th, y_test_th = train_test_split( 
    #        X_th, y_th, test_size=0.33,random_state=0)
     
    
    #%%
    # Use N-fold cross validation (67% training, 33% test)
    
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    
    fold = 5
    
    k_fold = KFold(n_splits=fold, shuffle=True, random_state=3)
    #k_fold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=3)
    
    ens_acc = {}
    for ens in range(20):
    
        #print(model.summary())
            
        cv_acc = []
    
    
        for i in range(5,10):
            
            K.backend.clear_session()
            tf.set_random_seed(ens)
            sess = tf.Session(graph=tf.get_default_graph())
            K.backend.set_session(sess)
            model = Sequential()
        
            model.add(K.layers.Conv1D(32, 8,strides=8, border_mode='same',input_shape=(1200,1), activation='relu'))
            model.add(K.layers.Conv1D(32, 5,strides=5, border_mode='same', activation='relu'))
            model.add(K.layers.Conv1D(32, 3,strides=3, border_mode='same', activation='relu'))
            model.add(K.layers.pooling.GlobalAveragePooling1D())
        
        
        
        #model.add(K.layers.Dense(128, activation='relu'))
        
            model.add(K.layers.Dense(n_classes, activation='softmax'))
            
            exp_num = 50+i*200
            
        
            accuracy=[]
            logs=[]
            ground_truth=[]
            predictions_ord=[]
            trains=[]
            tests=[]
            
            
            for k, (train, test) in enumerate(k_fold.split(X_exp, y_exp)):       
                    
                    #Save splits for later use
                    trains.append(train)
                    tests.append(test)
                    start_time = time.time()
                    #data augmentation to experimenal traning dataset
                    temp_x = X_exp[train]
                    temp_y = y_exp[train]
            
                        
                    exp_train_x,exp_train_y = exp_augdata(temp_x.T,exp_num,temp_y)
                    #combine theorectical and experimenal dataset for training
                    train_combine = np.concatenate((X_th,exp_train_x.T))
                  
                    train_dim = train_combine.reshape(train_combine.shape[0],1200,1)
                    train_y = np.concatenate((y_th,exp_train_y))
                    train_y_hot = enc.transform(train_y.reshape(-1,1))
                    
                    optimizer = K.optimizers.Adam()
                    
                    model.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['categorical_accuracy'])
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
                    
            #        reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
            #                                      patience=50, min_lr=0.00001)
            
            
                    test_x = X_exp[test]
                    test_x = test_x.reshape(test_x.shape[0],1200,1)
                    test_y = enc.transform(y_exp.reshape(-1,1))[test]
                    hist = model.fit(train_dim, train_y_hot, batch_size=BATCH_SIZE, nb_epoch=100,
                                     verbose=0, validation_data=(test_x, test_y), callbacks = [early_stop])
                    
                    #Compute model predictions
                    prediction=model.predict(test_x)
                    #Go from one-hot to ordinal...
                    prediction_ord=[np.argmax(element) for element in prediction]
                    predictions_ord.append(prediction_ord)        
                    
                    
                    #Produce ground_truth, each list element contains array with test elements on first column with respect to X_exp and 
                    ground_truth.append(np.concatenate([test.reshape(len(test),1),y_exp[test].reshape(len(test),1)],axis=1))
                    
                    #Compute loss and accuracy for each k validation
                    accuracy.append(model.evaluate(test_x, test_y, verbose=0))
                    
                    #Save logs
                    log = pd.DataFrame(hist.history)
                    logs.append(log)
                   
                    #Save models on current folder with names of 0 to 4
                    model.save('keras_model'+str(k)+'.h5')
        
        #%%
            accuracy = np.array(accuracy)   
            
            cv_accuracy = np.mean(accuracy[:,1])     
            print ('cross val accuracy', np.mean(accuracy[:,1]))  
            cv_acc.append(cv_accuracy)
        ens_acc[ens]=cv_acc
    ens_acc = np.array([list(v) for v in ens_acc.values()])
    
    ens_acc_mean = 100*np.mean(ens_acc,axis=0)
    
    ens_acc_std = 100*np.std(ens_acc,axis=0)
    acc[theor_num] = list(ens_acc_mean)
    err[theor_num] = list(ens_acc_std)



import numpy as np
s_0=np.concatenate((Runsim0exp0and1,Runsim0exp2to6,Runsim0exp7to10), axis=1)
s_100=np.concatenate((Runsim100exp0to4,Runsim100exp5to10),axis=1)
s_800=np.concatenate((Runsim800exp0to4,Runsim800exp5to10),axis=1)

import os
os.chdir(r"C:\Users\fovie\Documents\GitHub\XRD\Danny XRD check\Dimensionality Run")
np.savetxt("simulated_0.csv", s_0, delimiter=",")
np.savetxt("simulated_100.csv", s_100, delimiter=",")
np.savetxt("simulated_800.csv", s_800, delimiter=",")