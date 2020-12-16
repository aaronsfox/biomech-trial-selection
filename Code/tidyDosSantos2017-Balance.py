# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:09:07 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function tidied up the dos Santos et al. (2017) balance data files into
    individual participant folders for later access. After running once the code will 
    no longer work, and simply served as a way to initially manage the dataset.
    
"""

# %% Import packages

import os
import glob
import shutil

# %% Tidy dataset

#Navigate to dataset folder
os.chdir('..\\DosSantos2017-BalanceDataset')

#Identify subject list based on filenames
#Get file list of static trials
staticTrials = list()
for file in glob.glob('*static.txt'):
    staticTrials.append(file)
    
#Loop through static trials and get participant list
subList = list()
for ii in range(len(staticTrials)):
    #Split string and append to list
    subList.append(staticTrials[ii].split('static')[0])
    
#Loop through participants and allocate files to folder
for ii in range(1,len(subList)):
    
    #Create blank list for files
    subFiles = list()
    
    #Get files for current subject
    for file in glob.glob(subList[ii]+'*'):
        subFiles.append(file)
        
    #Create directory for current subject
    os.mkdir(subList[ii])
    
    #Loop through file list
    #Move if an angle or grf file
    #Delete if a marker or static file
    for ff in range(len(subFiles)):
        if 'ang.txt' in subFiles[ff] or 'grf.txt' in subFiles[ff]:
            shutil.move(subFiles[ff],subList[ii]+'\\'+subFiles[ff])
        elif 'mkr.txt' in subFiles[ff] or 'static.txt' in subFiles[ff]:
            os.remove(subFiles[ff])
            
    
# %% ----- End of tidyDosSantos2017-Balance.py -----