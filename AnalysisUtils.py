import numpy as np
import pandas as pd 
import os

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt


ped   = list(range(10, 2000))   # to estimate the pedestal
rowin = list(range(3, 100004))   # all readout window
wf    = list(range(4000, 50000))  # waveform 
tail  = list(range(60500, 100000)) # to estimate the single p.e.


def defineRoI(df):
    
    if (df.shape[1]==50004):
        ped   = list(range(10, 2000))   # to estimate the pedestal
        rowin = list(range(3, 50004))   # all readout window
        wf    = list(range(4000, 20000))  # waveform 
        tail  = list(range(40500, 50000)) # to estimate the single p.e.

        
        print('Defined Region of Interest: ' )
        print('Pedestal region (ped)  : [', min(ped)  , ',' , max(ped)  , ']'  ) 
        print('Readout window (rowin) : [', min(rowin), ',' , max(rowin), ']'  ) 
        print('Waveform (wf)          : [', min(wf)   , ',' , max(wf)   , ']'  ) 
        print('Waveform tail (tail)   : [', min(tail) , ',' , max(tail) , ']'  ) 

    else:   
        
        ped   = list(range(10, 2000))   # to estimate the pedestal
        rowin = list(range(3, 100004))   # all readout window
        wf    = list(range(4000, 50000))  # waveform 
        tail  = list(range(60500, 100000)) # to estimate the single p.e.

        
        print('Defined Region of Interest: ' )
        print('Pedestal region (ped)  : [', min(ped)  , ',' , max(ped)  , ']'  ) 
        print('Readout window (rowin) : [', min(rowin), ',' , max(rowin), ']'  ) 
        print('Waveform (wf)          : [', min(wf)   , ',' , max(wf)   , ']'  ) 
        print('Waveform tail (tail)   : [', min(tail) , ',' , max(tail) , ']'  ) 
        
        
    return ped, rowin, wf, tail




#
# Routine to prepare datasets: check for possible saturation and compute, subtract pedestal 
##
def prepare_dataset(_df):

    _df = _df.copy()        
    print('preparing dataframe for channel : ') 

    ped, rowin, wf, tail = defineRoI(_df)

    _df = do_reindex(_df)
    _df = convert_time(_df)
    #_df = define_channel(_df)
    #_df = do_reindex(_df)
    _df = flag_saturated(_df, 65536)
    _df = compute_pedestal(_df)
    _df = subtract_pedestal(_df)
    _df = remove_noise(_df) 
    _df = has_signal_new(_df)
    _df = compute_singlepe(_df)
    #_df = select_singlepe(_df)
    #_df = tagGoodwf(_df)        

    print('done!')

    return _df



def read_file(file_name_list, start, stop):
    df_list =[]    
    _df = pd.DataFrame()

    for f in file_name_list[start:stop]:
        print(f)
        df_tmp = pd.read_csv(f, sep='\t', header=None) 
        
        df_list.append(df_tmp)

    return pd.concat(df_list, axis=0)


def do_reindex(df):
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    return df


def convert_time(df):
    df = df.copy()
    df[2] =  pd.to_datetime(df[2], format='%Y/%m/%d %H:%M:%S.%f')
    return df

#
# Flag the saturated waveform
###
def flag_saturated(df, val=10000):
    df=df.copy()
    df['Saturated']=(df[rowin].max(axis=1) >= val)
    return df


def compute_pedestal(df):
    df=df.copy()
    df['Pedestal']=df[ped].sum(axis=1)/len(ped)
    return df

def subtract_pedestal(df):
    df = df.copy()
    df[rowin] = df[rowin].subtract(df['Pedestal'], axis=0)
    return df

def remove_noise(df):
    df=df.copy()
    df[rowin]= df[rowin].rolling(window=250,  axis=1).mean()
    return df


def has_signal_new(df):
    df = df.copy()
    
    df_sig = df.apply(lambda x: find_signal(x, wf), axis=1)    
    df = pd.concat([df, df_sig], axis =1)

    return df


def find_signal(x, myrange):
    
    x = x[min(myrange) : max(myrange)]

    peaks, properties = find_peaks(x, height=[15,6000], width=10)
    peaks = peaks+min(myrange)
    
    npeaks = len(peaks)  
    if (npeaks > 0):
        height = properties['peak_heights'][0]
        width  = properties['widths'][0]
        xlow   = int(properties['left_ips'][0])
        xhigh  = int(properties['right_ips'][0])
        area   = x[xlow :xhigh].sum() 
    else :
        height = 0
        width  = 0
        area   = 0

    return pd.Series([(npeaks>0), height, area], index=['hasSignal', 'signal height', 'signal area'])


def find_singlePE(x, myrange):

    x = x[min(myrange) : max(myrange)]
    
    #peaks, properties = find_peaks(x, height=[2000,8000], width=600, distance=4000)  # first test for calib ch2
    #peaks, properties = find_peaks(x, height=[2000,6000], width=600, distance=4000)  # first test for calib ch2
    #peaks, properties = find_peaks(x, prominence=2000, width=600, distance=4000)  # first test for calib ch2
    peaks, properties = find_peaks(x, height=[2000,4500], prominence=2000, width=1000, distance=4000) 
    widths, width_heights, leftwidth_ips, rightwidth_ips = peak_widths(x, peaks, rel_height=0.85)

    
    peaks = peaks+min(myrange)
    
    npeaks = len(peaks)  
    if (npeaks > 0):
        #height = properties['peak_heights'][0]
        height = properties['prominences'][0]
        #width  = properties['widths'][0]
        width  = widths[0]
        #xlow   = int(properties['left_ips'][0])
        #xhigh  = int(properties['right_ips'][0])
        xlow   = int(leftwidth_ips[0])
        xhigh  = int(rightwidth_ips[0])
        area   = x[xlow :xhigh].sum() 
    else :
        height = 0
        width  = 0
        area   = 0
        
    #return pd.Series([npeaks, height, width, area], index=['n pe', 'pe height', 'pe width', 'pe area'])
    return pd.Series([npeaks, height, width, area], index=['n pe', 'pe height', 'pe width', 'pe area'])




def compute_singlepe(df):
    df = df.copy()

    df_pe = df.apply(lambda x: find_singlePE(x, rowin), axis=1)    
    df = pd.concat([df, df_pe], axis =1)
    
    return df


def select_singlepe(df):
    df = df.copy()

    X_pe=df.loc[(df['pe height']>0)& (df['pe width']>0),['pe height', 'pe width']].values

    #X_pe=df.loc[(df['Saturated'] == False ) & 
    #            #(df['hasSignal'] == True ) &
    #            (df['pe height']>0)&
    #            (df['pe width']>0),['pe height', 'pe width']].values


    if (len(X_pe)>0):

        mu_pe, cov_pe = estimate_gaus_param(X_pe,True)

        df['spe 1sig'] = df.apply(lambda x: select_wf(x[['pe height','pe width']], mu_pe, cov_pe, 1), axis=1)
        df['spe 2sig'] = df.apply(lambda x: select_wf(x[['pe height','pe width']], mu_pe, cov_pe, 2), axis=1)

    else:
        df['spe 1sig'] = False
        df['spe 2sig'] = False

    return df


def estimate_gaus_param(X, multivar=False):
    mean = np.mean(X, axis=0)
    
    if multivar:
        cov = 1/float(len(X)) * np.dot( (X - mean).T , X-mean)
    else:
        cov = np.diag(np.var(X, axis=0))
    return mean,cov


def select_wf(xy, mean, cov, n_sigma):
 
    Z = multivariate_normal.pdf( xy , mean=mean, cov=cov)
    #print(Z)
    
    sigma = np.sqrt(np.diag(cov))
    #limit = n_sigma * sigma     
    limit = mean + n_sigma * sigma     
        
    thrsld = multivariate_normal.pdf( limit, mean=mean, cov=cov)
    #print(thrsld)
    
    return Z > thrsld
