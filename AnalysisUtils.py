import numpy as np
import pandas as pd 
import os

#from scipy.stats import norm
#from scipy.stats import multivariate_normal
#from scipy.signal import find_peaks

#import matplotlib.pyplot as plt

ped   = list(range(10, 2000))   # to estimate the pedestal
rowin = list(range(3, 50004))   # all readout window
wf    = list(range(4000, 3000))  # waveform 
tail  = list(range(40500, 50000)) # to estimate the single p.e.

def defineRoI():
    
    print('Defined Region of Interest: ' )
    print('Pedestal region (ped)  : [', min(ped)  , ',' , max(ped)  , ']'  ) 
    print('Readout window (rowin) : [', min(rowin), ',' , max(rowin), ']'  ) 
    print('Waveform (wf)          : [', min(wf)   , ',' , max(wf)   , ']'  ) 
    print('Waveform tail (tail)   : [', min(tail) , ',' , max(tail) , ']'  ) 

    return ped, rowin, wf, pe

def read_file(file_name_list):
    df_list =[]    
    _df = pd.DataFrame()

    for f in file_name_list:
        #print(f)
        df_tmp = pd.read_csv(f, sep='\t', header=None) 
        
        df_list.append(df_tmp)

    return pd.concat(df_list, axis=0)