import pandas as pd
import numpy as np
import glob
import copy

#%% Read File
all_file = []
SubjectID = []
gt = []
or_Adls = []
or_falls = []
Adl = pd.DataFrame()
fall = pd.DataFrame()

path = r''
ADL_address = (glob.glob(path+'\All_Sub\*\ADL\*.csv'))
fall_address = (glob.glob(path+'\All_Sub\*\Fall\*.csv')) 
 
# Labeling
for i in range((len(ADL_address))):
    Adl = pd.read_csv(ADL_address[i])
    or_Adls.append(Adl.drop(columns = ["TimeStamp(s)","FrameCounter"]))
    gt.append(0)
    
for i in range((len(fall_address))):
    fall = pd.read_csv(fall_address[i])
    or_falls.append(fall.drop(columns = ["TimeStamp(s)","FrameCounter"]))
    gt.append(1)

Imp_Lab = pd.read_excel(r'.xlsx',engine ='openpyxl')
su_a = np.load(r'.npy')
su_f = np.load(r'.npy')

#%% Gaussian Noise
def gaussian_noise(data, mag = -2):
    BS, L = data.shape
    sigma = np.std(data, axis = 1)
    sigma = np.reshape(sigma,(BS,1))
    n = np.random.randn(BS,L)
    noise = 0.25*(1/(1+np.exp(-mag)))*sigma*n
    return data + noise
    
#%% Magnitude Scale
def magnitude_scale(x, mag):
    BS, L, C = x.shape
    bs = np.random.rand(BS)
    bs = np.reshape(bs, (BS,1,1))
    strength = (1/(1+np.exp(-mag)))*(-0.5*bs+1.25)
    strength = np.reshape(strength, (BS,1,1))
    return x*strength

#%% Define the ADL windows 
lon = 50
Fall_onset_frame = np.array(Imp_Lab["Fall_onset_frame"])
Impact_frame = np.array(Imp_Lab['Fall_impact_frame'])

# Each ADL instance is divided into ten parts equally. Then, we captured a window with a size of 50 frames (0.5s) from each part. 
def ADL_window(Adls):
    sliding_adl_data = []
    sliding_adl_sub = []
    sliding_adl_gt = []

    for i in range(len(Adls)):
        for j in range(10):
            if int(len(Adls[i]))/10*j+60 <= len(Adls[i]):
                sliding_adl_data.append(Adls[i][int(len(Adls[i])/10)*j+10:int(len(Adls[i])/10)*j+60])
                sliding_adl_gt.append(0)
                sliding_adl_sub.append(su_a[i])
                
    return sliding_adl_data, sliding_adl_gt, sliding_adl_sub  
    
#%%  Define the fall windows of training dataset   
def fall_window_training(falls, Fall_onset_frame, Impact_frame, lon, su_f): 
    preadl_data = []   
    preadl_gt = []
    preadl_sub = []
 
    prefall_data = []
    prefall_gt = []
    prefall_sub = []        
    # Sliding window
    for i in range(len(falls)):
        for j in range(150):
            if lon+10*j < Fall_onset_frame[i] :
                preadl_data.append(falls[i][10*j:lon+10*j])
                preadl_gt.append(0)
                preadl_sub.append(su_f[i])
            elif Impact_frame[i] >= lon+10*j > Fall_onset_frame[i] and (lon+10*j - Fall_onset_frame[i] >= 5):
                prefall_data.append(falls[i][10*j:lon+10*j])
                prefall_gt.append(1)
                prefall_sub.append(su_f[i])
    return  preadl_data, preadl_gt, preadl_sub, prefall_data, prefall_gt, prefall_sub
    
#%% fall window testing
def fall_window_testing(falls, Fall_onset_frame, su_f, Impact_frame, lon):
    test_data = []
    test_sub = []
    test_gt = [] 
    for i in range(len(falls)):
        for j in range(10): 
            if lon +10*j < Fall_onset_frame[i] :
                test_data.append(falls[i][10*j:lon+10*j])
                test_gt.append(0)
                test_sub.append(su_f[i])
        # The first 3 pre-impact windows reaching over the fall onset moment in each fall instance were collected        
        for j in range(100):             
           if (lon+10*j - Fall_onset_frame[i] >= 5):
               test_data.append(falls[i][10*j:lon+10*j])
               test_gt.append(1)
               test_sub.append(su_f[i]) 
               test_data.append(falls[i][10*(j+1):lon+10*(j+1)])
               test_gt.append(1)
               test_sub.append(su_f[i])
               test_data.append(falls[i][10*(j+2):lon+10*(j+2)])
               test_gt.append(1)
               test_sub.append(su_f[i])
               break
           
    return test_data, test_gt, test_sub
   
#%% main
def main(Adls, falls):
    sliding_adl_data, sliding_adl_gt, sliding_adl_sub= ADL_window(Adls)
    preadl_data, preadl_gt, preadl_sub, prefall_data, prefall_gt, prefall_sub = fall_window_training(falls, Fall_onset_frame, Impact_frame, lon, su_f)
    test_data, test_gt, test_sub = fall_window_testing(falls, Fall_onset_frame, su_f, Impact_frame, lon)
    
    prefall_data = np.tile(prefall_data,(6, 1, 1))
    prefall_gt = np.tile(prefall_gt,(6))
    prefall_sub = np.tile(prefall_sub,(6))
    
    final_train_data = np.concatenate((np.array(sliding_adl_data), np.array(preadl_data), np.array(prefall_data)))
    final_train_gt = np.concatenate((np.array(sliding_adl_gt), np.array(preadl_gt), np.array(prefall_gt)))
    final_train_sub = np.concatenate((np.array(sliding_adl_sub), np.array(preadl_sub), np.array(prefall_sub)))
    final_test_data = np.concatenate((np.array(sliding_adl_data), np.array(test_data)))
    final_test_gt = np.concatenate((np.array(sliding_adl_gt), np.array(test_gt)))
    final_test_sub = np.concatenate((np.array(sliding_adl_sub), np.array(test_sub))) 
    return final_train_data, final_train_gt, final_train_sub, final_test_data, final_test_gt, final_test_sub
#%% load        
original_train_data, original_train_gt, original_train_sub, original_test_data, original_test_gt, original_test_sub = main(or_Adls, or_falls)
original_train_data_norm = copy.deepcopy(original_train_data)  

#%%
gaussian_train_data = copy.deepcopy(original_train_data)
for i in range(len(gaussian_train_data[0,0,:])):
    gaussian_train_data[:,:,i] = gaussian_noise(gaussian_train_data[:,:,i], mag = -0.5)

mag_train_data = copy.deepcopy(original_train_data)
mag_train_data = magnitude_scale(mag_train_data, mag = -0.5)        

final_train_data =  np.concatenate((np.array(original_train_data), np.array(gaussian_train_data), np.array(mag_train_data)))     
final_train_gt = np.concatenate((np.array(original_train_gt), np.array(original_train_gt), np.array(original_train_gt))) 
final_train_sub = np.concatenate((np.array(original_train_sub), np.array(original_train_sub), np.array(original_train_sub)))   
                                    
path = r''  
np.save(path+'\\KFall_final_train_data',final_train_data)
np.save(path+'\\KFall_final_train_gt',final_train_gt)
np.save(path+'\\KFall_final_train_sub',final_train_sub) 

np.save(path+'\\KFall_original_train_data',original_train_data)
np.save(path+'\\KFall_original_train_gt',original_train_gt)
np.save(path+'\\KFall_original_train_sub',original_train_sub)

np.save(path+'\\KFall_final_test_data',original_test_data)
np.save(path+'\\KFall_final_test_gt',original_test_gt)
np.save(path+'\\KFall_final_test_sub',original_test_sub)
