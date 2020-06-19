import numpy as np
import pandas as pd 
import scipy as sp
import scipy.io
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import glob
import fnmatch
import re
import argparse

import sys, getopt


from AnalysisUtils import *

def main(start, stop):

    #base_path = '/eos/user/s/sbordoni/SiPM_XArapucafiles/calib15june-ch2/data'
    base_path = '/Users/bordoni/protoDUNE/XeDoping/SiPM-XArapuca_save/data/calib_june15_ch3'


    file_path = os.path.join(base_path,'C3*.dat')

    file_name_list =  glob.glob(file_path) 
    #print(file_name_list)

    df_wf_ch2 = read_file(file_name_list, start, stop)

    df_wf_ch2_proc = prepare_dataset(df_wf_ch2)

    outfilename1='calib15june-ch3_files.csv'
    df_wf_ch2_proc.to_csv(outfilename1)

    outfilename='calib15june-ch3_'+ str(start)+'-'+str(stop)+'files.csv'
    df_wf_ch2_proc.to_csv(outfilename)

    print('done!')
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--first', type = int, required=False, default = 0, help='first file to read')
    parser.add_argument('-l', '--last', type = int, required=False, default = -1, help='last file to read')
    args = parser.parse_args()
    
    main(args.first, args.last) 


