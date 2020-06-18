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











if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--first', type = int, required=False, default = 0, help='files to read')
    #parser.add_argument('-l', '--last', type = int, required=False, default = -1, help='last element of the runlist to run the code')
    args = parser.parse_args()
    
    main(args.first) 
