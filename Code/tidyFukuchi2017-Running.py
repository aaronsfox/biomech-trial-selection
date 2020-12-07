# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:43:19 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function tidied up the Fukuchi et al. (2017) running C3D data files into
    individual participant folders for later access. After running once the code will 
    no longer work, and simply served as a way to initially manage the dataset.
    
"""

# %% Import packages

import os
import glob
import shutil

# %% Tidy dataset

#Navigate to dataset folder
os.chdir('..\\Fukuchi2017-RunningDataset')

#Identify subject list based on filenames
#Get file list of static trials
staticTrials = list()
for file in glob.glob('*static.c3d'):
    staticTrials.append(file)
    
#Loop through static trials and get participant list
subList = list()
for ii in range(len(staticTrials)):
    #Split string and append to list
    subList.append(staticTrials[ii].split('static.c3d')[0])
    
#Loop through participants and allocate files to folder
for ii in range(1,len(subList)):
    
    #Create blank list for files
    subFiles = list()
    
    #Get files for current subject
    for file in glob.glob(subList[ii]+'*'):
        subFiles.append(file)
        
    #Create directory for current subject
    os.mkdir(subList[ii])
    
    #Loop through file list and move
    for ff in range(len(subFiles)):
        shutil.move(subFiles[ff],subList[ii]+'\\'+subFiles[ff])
    
# %% ----- End of tidyFukuchi2017-Running.py -----
    