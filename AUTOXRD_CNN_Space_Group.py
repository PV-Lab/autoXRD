"""
AUTO-XRD V 0.9

Code to perform classification of XRD patterns for various spcae-group using 
physics-informed data augmentation and all convolutional neural network (a-CNN).
Code to plot class activation maps from a-CNN

@authors: Felipe Oviedo and Danny Zekun Ren

Code is under MIT license, please cite any use of the code as explained 
in README, in the GitHub repository.

"""

#!usr/bin/env python  
#-*- coding: utf-8 -*-  

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
#from keras models import Sequential
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping

##define multiple machine learning algorithms


import tensorflow as tf

K.backend.clear_session()

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

tf.reset_default_graph()
# Parameters

BATCH_SIZE=128

# Network Parameters
n_input = 1200 # MNIST data input (img shape: 28*28)
n_classes = 7 # MNIST total classes (0-9 digits)
#drop_out = 0.2 # Dropout, probability to keep units
filter_size = 2
kernel_size = 10

enc = OneHotEncoder(sparse=False)

model = Sequential()

model.add(K.layers.Conv1D(32, 12,strides=8, border_mode='same',input_shape=(1200,1), activation='relu'))
model.add(K.layers.Conv1D(32, 8,strides=5, border_mode='same', activation='relu'))
model.add(K.layers.Conv1D(32, 5,strides=3, border_mode='same', activation='relu'))
model.add(K.layers.pooling.GlobalAveragePooling1D())

print(model.summary())

model.add(K.layers.Dense(n_classes, activation='softmax'))
  
        

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

#data normalization from 0 to 1 for double column dataframe, returns single column array
def normdata(data):
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1//2])
    for i in range(w1//2):
        ndata[:,i]=(data[:,2*i+1]-min(data[:,2*i+1]))/(max(data[:,2*i+1])-min(data[:,2*i+1]))
    return ndata
#data normalization from 0 to 1 for single column dataframe
def normdatasingle(data):
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1])
    for i in range(w1):
        ndata[:,i]=(data[:,i]-min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
    return ndata

#data augmendatation for simulated XRD spectrum
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
#data augmendatation for experimental XRD spectrum
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
        #nomrlization
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

#extracting exprimental data
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


#os.chdir(r"/Users/danny/Dropbox/work/Campaign/XRD space group/Danny XRD check/Theoretical")
os.chdir(r"C:\Users\fovie\Dropbox (MIT)\MIT Research\XRD ML\XRD space group\Danny XRD check\Theoretical")
#Produce lists with sample names
excluded="test.ASC"
directory=[]
for dir,dirs, files in os.walk("."):
    
    directory.extend(dirs)

spectra_th=[]
for k in directory:
    spectra_th.append(spectra_list(k,excluded))

#Create dictionary of dataframes for each symmetry group    
x = directory[:7]
grp = {}
for indx,vals in enumerate(x):
    grp['%s' %vals] = group(spectra_th,indx)

theor=pd.concat(grp,axis=1)
pd.DataFrame(theor).to_csv(r'C:\Users\fovie\Documents\GitHub\AUTO-XRD\theor.csv')

pd.Series(theor.columns.get_level_values(0)).to_csv(r'C:\Users\fovie\Documents\GitHub\AUTO-XRD\label_theo.csv')

aa = list(theor.columns.get_level_values(0))

theor_arr=theor.values

#%Normalize data
ntheor = normdata (theor_arr)

#os.chdir(r"/Users/danny/Dropbox/work/Campaign/XRD space group/Danny XRD check/Experimental")
os.chdir(r"C:\Users\fovie\Dropbox (MIT)\MIT Research\XRD ML\XRD space group\Danny XRD check\Experimental")
#Produce lists with sample names
directory2=[]
for dir,dirs, files in os.walk("."):
    directory2.extend(dirs)

spectra_exp=[]
for k in directory2:
    spectra_exp.append(spectra_list2(k))


#Create dictionary of dataframes for each symmetry group    
x = directory2;
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

#%%
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
num = 2000

augd,pard,crop_augd = augdata(ntheor,num,theor,theor_min,theor_max)    


#convert theorectical label from space group to numbers
label_t=np.zeros([len(pard),])
for i in range(len(pard)):
    
    temp = pard[i]
    label_t[i]=directory.index(temp[0])
    
pd.DataFrame(directory).to_csv(r'C:\Users\fovie\Documents\GitHub\AUTO-XRD\encoding.csv')

 #convert experimental label from space group to numbers       

par_exp = list(exp.columns.values)

label_exp=np.zeros([len(par_exp)//2,])

for i in range(len(par_exp)//2):
    
    temp = par_exp[2*i]
    label_exp[i]=directory2.index(temp[0])
    
#input the experimetal data points       
exp_num =86 

#train and test split for the experimental data
X_exp = np.transpose(post_exp[:,0:exp_num])
y_exp = label_exp[0:exp_num]

#y_exp_hot = enc.fit_transform(y_exp.reshape(-1,1))
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

accuracy_exp = np.empty((fold,1))
accuracy_exp_b = np.empty((fold,1)) 
accuracy_exp_r1 = np.empty((fold,1)) 
accuracy_exp_p1 = np.empty((fold,1))
accuracy_exp_r2 = np.empty((fold,1))  
accuracy_exp_p2 = np.empty((fold,1))
f1=np.empty((fold,1))
f1_m=np.empty((fold,1))

#k_fold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=3)


accuracy=[]
logs=[]
ground_truth=[]
predictions_ord=[]
trains=[]
tests=[]
trains_combine=[]
trains_y=[]
     

for k, (train, test) in enumerate(k_fold.split(X_exp, y_exp)):

        
        #Save splits for later use
        trains.append(train)
        tests.append(test)
        start_time = time.time()
        #data augmentation to experimenal traning dataset
        temp_x = X_exp[train]
        temp_y = y_exp[train]
        exp_train_x,exp_train_y = exp_augdata(temp_x.T,5000,temp_y)
        #combine theorectical and experimenal dataset for training
        train_combine = np.concatenate((X_th,exp_train_x.T))
        trains_combine.append(train_combine)
        
        
        K.backend.clear_session()


# Parameters

        BATCH_SIZE=128
        
        # Network Parameters
        n_input = 1200 # MNIST data input (img shape: 28*28)
        n_classes = 7 # MNIST total classes (0-9 digits)
        #drop_out = 0.2 # Dropout, probability to keep units
        filter_size = 2
        kernel_size = 10

        enc = OneHotEncoder(sparse=False)
      
        train_dim = train_combine.reshape(train_combine.shape[0],1200,1)
        train_y = np.concatenate((y_th,exp_train_y))
        trains_y.append(train_y)
        train_y_hot = enc.fit_transform(train_y.reshape(-1,1))
        
        model = Sequential()

        model.add(K.layers.Conv1D(32, 8,strides=8, padding='same',input_shape=(1200,1), activation='relu'))
        model.add(K.layers.Conv1D(32, 5,strides=5, padding='same', activation='relu'))
        model.add(K.layers.Conv1D(32, 3,strides=3, padding='same', activation='relu'))
        model.add(K.layers.pooling.GlobalAveragePooling1D())
        model.add(K.layers.Dense(n_classes, activation='softmax'))
        
        
        
        optimizer = K.optimizers.Adam()
        
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        
        #early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        
#        reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
#                                      patience=50, min_lr=0.00001)


        test_x = X_exp[test]
        test_x = test_x.reshape(test_x.shape[0],1200,1)
        test_y = enc.fit_transform(y_exp.reshape(-1,1))[test]
        hist = model.fit(train_dim, train_y_hot, batch_size=BATCH_SIZE, nb_epoch=100,
                         verbose=1, validation_data=(test_x, test_y))
        #callbacks = [early_stop]
        #Compute model predictions
        prediction=model.predict(test_x)
        #Go from one-hot to ordinal...
        prediction_ord=[np.argmax(element) for element in prediction]
        predictions_ord.append(prediction_ord)
        
        #Accuracy, recall, precision and F1
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
       
        #Save models on current folder with names of 0 to 4
        model.save('keras_model'+str(k)+'.h5')

#
accuracy = np.array(accuracy)        
print ('Mean Cross-val accuracy', np.mean(accuracy[:,1]))    

#%%
#Function, plot the cam and spectra of certain set of samples
    
from keras.models import load_model
#from vis.visualization import visualize_cam

#Function to plot CAM of a certain model, subset of test elements, and spectra array with all experimental data (non-augmented)
def get_cam(model_name, elements, spectra):
    
    model = load_model(model_name)        
    gap_weights = model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=model.input, 
                    outputs=(model.layers[-3].output, model.layers[-1].output))
    
    features, results = cam_model.predict(spectra[elements].reshape(spectra[elements].shape[0],1200,1))
    cam_outputs=np.zeros([10,1])
    
# check the prediction for 10 test images
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
    #ax.set_title("CAM of Single Pattern")
    print()
    
    if len(y2) != 0:
        ax2.plot(np.linspace(10,69.96,1200),y2)
        ax2.set_xlabel(r"$2 \theta$ (Degrees)")
        ax2.set_title("Pattern")
        
    
    fig.suptitle(title, fontsize=16)
    
        

def find_incorrects(ground_truth,predictions_ord):
    #Find corrects and incorrects in all models ran, and compare them to ground-truth
    
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


#%%

#Compute incorrects and comparision
corrects, incorrects=find_incorrects(ground_truth,predictions_ord)

#Get dataframe of all incorrects and dataframe of all corrects
corrects_df = pd.concat([r for r in corrects], ignore_index=False, axis=0)
incorrects_df = pd.concat([r for r in incorrects], ignore_index=False, axis=0)

#Get the cam for the trained examples, for each class we average the cam of all trained examples
#Trains refers to the elements in X_exp used for training
cam_outputs=get_cam('keras_model4.h5', trains[4], X_exp)
cam_df=pd.DataFrame(cam_outputs)
cam_df=cam_df.iloc[1:]
cam_df['Label']=y_exp[trains[4]]

##CAM with all augmented data
rng=range(0,7000)
cam_outputs2=get_cam('keras_model4.h5', rng, train_combine)
cam_df2=pd.DataFrame(cam_outputs2)
cam_df2=cam_df2.iloc[1:]
cam_df2['Label']=train_y

#Now we focus on the incorrectly labelled cam's
incorrects_filtered=incorrects_df[incorrects_df.Model=='keras_model4.h5']


cam_inc=get_cam('keras_model4.h5', [int(element) for element in incorrects_filtered.index], X_exp)
cam_inc=pd.DataFrame(cam_inc)
cam_inc=cam_inc.iloc[1:]    

#Cams
cam_df=cam_df2

#Get the mean class maps for each class, for model keras_model0.h5
cam_filtered=cam_df[cam_df.Label==0]
means_0=cam_filtered.mean()
means_0=means_0.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==1]
means_1=cam_filtered.mean()
means_1=means_1.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==2]
means_2=cam_filtered.mean()
means_2=means_2.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==3]
means_3=cam_filtered.mean()
means_3=means_3.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==4]
means_4=cam_filtered.mean()
means_4=means_4.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==5]
means_5=cam_filtered.mean()
means_5=means_5.iloc[:-1]
cam_filtered=cam_df[cam_df.Label==6]
means_6=cam_filtered.mean()
means_6=means_6.iloc[:-1]


#Plot the cam for  each class, we leave the spectra as none
plot_cam(means_0,'Average CAM for Class 0, trained model4.h5')
plot_cam(means_1,'Average CAM for Class 1, trained model4.h5')
plot_cam(means_2,'Average CAM for Class 2, trained model4.h5')
plot_cam(means_3, 'Average CAM for Class 3, trained model4.h5')
plot_cam(means_4, 'Average CAM for Class 4, trained model4.h5')
plot_cam(means_5,'Average CAM for Class 5, trained model4.h5')
plot_cam(means_6,'Average CAM for Class 6, trained model4.h5')

#Plot the cam for the incorrect examples
plot_cam(cam_inc.iloc[0,:],'Incorrect: true class 2, predicted is 6', X_exp[int(incorrects_filtered.index[0])])
plot_cam(cam_inc.iloc[1,:],'Incorrect: true class 4, predicted is 0', X_exp[int(incorrects_filtered.index[1])])
plot_cam(cam_inc.iloc[2,:],'Incorrect: true class 5, predicted is 4', X_exp[int(incorrects_filtered.index[2])])
plot_cam(cam_inc.iloc[3,:],'Incorrect: true class 6, predicted is 1', X_exp[int(incorrects_filtered.index[3])])   
plot_cam(cam_inc.iloc[4,:],'Incorrect: true class 6, predicted is 3', X_exp[int(incorrects_filtered.index[4])])  


#Plot correctly classied 

corrects_filtered=corrects_df[corrects_df.Model=='keras_model4.h5']
cam_cor=get_cam('keras_model4.h5', [int(element) for element in corrects_filtered.index], X_exp)
cam_cor=pd.DataFrame(cam_cor)
cam_cor=cam_cor.iloc[1:]

plot_cam(cam_cor.iloc[-1,:],'Incorrect: true class 6, predicted is 6', X_exp[int(corrects_filtered.index[-1])])
plot_cam(cam_cor.iloc[-2,:],'Incorrect: true class 6, predicted is 6', X_exp[int(corrects_filtered.index[-2])])
plot_cam(cam_cor.iloc[-3,:],'Incorrect: true class 6, predicted is 6', X_exp[int(corrects_filtered.index[-3])])
plot_cam(cam_cor.iloc[-4,:],'Incorrect: true class 6, predicted is 6', X_exp[int(corrects_filtered.index[-4])])


plot_cam(means_6,'Mean for Class 6, trained model4.h5')

