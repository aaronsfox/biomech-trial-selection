# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:23:37 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function compiles the processed data from the Fukuchi et al. (2017)
    treadmill running dataset, and works through the sampling/simulation process
    of selecting gait cycles and comparing the outputs.
    
    Specific notes on analysis processes are outlined in the comments.
    
"""

# %% Import packages

import opensim as osim
import os
import pandas as pd
import btk
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import spm1d
from statsmodels.stats.anova import AnovaRM
from scipy import stats
# import pickle
import _pickle as cPickle
import bz2
import tqdm
import time

# %% Define functions

#Convert opensim array to python list
#For use in other functions
def osimToList(array):
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp

#Convert opensim storage object to dataframe
def stoToDF(osimSto):
    """
    
    Parameters
    ----------
    osimSto: opensim storage object previously loaded

    Returns
    -------
    df: pandas data frame
    
    """

    #Get column labels
    labels = osimToList(osimSto.getColumnLabels())    
    #Set time values
    time = osim.ArrayDouble()
    osimSto.getTimeColumn(time)
    time = osimToList(time)    
    #Allocate space for data
    data = []
    #Loop through columns and get data
    for i in range(osimSto.getSize()):
        temp = osimToList(osimSto.getStateVector(i).getData())
        temp.insert(0, time[i])
        data.append(temp)
    #Convert to dataframe
    df = pd.DataFrame(data, columns=labels)
    
    return df

#Identify foot strike timings
def footStrikeID(c3dFilename):
    
    #Inputs:
    #   c3dFilename: name of c3d file to extract foot strikes from

    #Returns:
    #   leftFS: list of left foot strike times
    #   rightFS: list of right foot strike times

    #Load in the c3d data via btk
    c3d = btk.btkAcquisitionFileReader()
    c3d.SetFilename(c3dFilename)
    c3d.Update()
    c3dAcq = c3d.GetOutput()
    
    #Extract the force platform data
    fp = btk.btkForcePlatformsExtractor()
    fp.SetInput(c3dAcq)
    fp.Update()
    
    #Get the wrenchs for position and force data
    grw = btk.btkGroundReactionWrenchFilter()
    grw.SetInput(fp.GetOutput())
    grw.Update()
    
    #Get the force and cop data from the one plate
    f = grw.GetOutput().GetItem(0).GetForce().GetValues()
    cop = grw.GetOutput().GetItem(0).GetPosition().GetValues()        
    
    #Check for any gaps in force signal prior to filtering
    f_interp = np.empty((len(f),3))
    cop_interp = np.empty((len(cop),3))
    #Loop through axes
    for aa in range(3):
        
        #Force data
        if np.sum(np.isnan(f[:,aa])) > 0:
            
            #Create a timestamp for the signal
            t = np.linspace(0,1/c3dAcq.GetAnalogFrequency()*(len(f)-1),len(f))
            
            #Remove nans from data array and corresponding times
            yi = f[:,aa][~np.isnan(f[:,aa])]
            xi = t[~np.isnan(f[:,aa])]
            
            #Interpolate data to fill gaps
            cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
            f_interp[:,aa] = cubicF(np.linspace(t[0], t[-1], len(t)))
            
        else:
            
            #Set data values to the original array
            f_interp[:,aa] = f[:,aa]
            
        #COP data
        if np.sum(np.isnan(cop[:,aa])) > 0:
            
            #Create a timestamp for the signal
            t = np.linspace(0,1/c3dAcq.GetAnalogFrequency()*(len(cop)-1),len(cop))
            
            #Remove nans from data array and corresponding times
            yi = cop[:,aa][~np.isnan(cop[:,aa])]
            xi = t[~np.isnan(cop[:,aa])]
            
            #Interpolate data to fill gaps
            cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
            cop_interp[:,aa] = cubicF(np.linspace(t[0], t[-1], len(t)))
            
        else:
            
            #Set data values to the original array
            cop_interp[:,aa] = cop[:,aa]
    
    #Filter force signals using 10Hz as per Fukuchi paper
    #Set cut-off
    fc = 10
    #Extract sampling rate of plate
    fs = c3dAcq.GetAnalogFrequency()
    #Define filter (2nd order filter for dual pass 4th order)
    b,a = butter(2, fc / (fs / 2.0), btype = 'low', analog = False)
    #Loop through 3D axes and filter
    f_filt = np.empty((len(f_interp),3))
    cop_filt = np.empty((len(cop_interp),3))
    for aa in range(3):
        #Force data
        f_filt[:,aa] = filtfilt(b, a, f_interp[:,aa])
        #COP data
        cop_filt[:,aa] = filtfilt(b, a, cop_interp[:,aa])
    
    #Convert any vertical force values below 20N threshold to zero
    #Get boolean of force threshold
    offForce = f_filt[:,1] < 20
    #Set any relevant data to zero
    f_filt[offForce,:] = 0
    cop_filt[offForce,:] = 0

    #Identify the foot strike timings based on a 20N threshold as per paper
    #Get indices where above threshold
    fpMask = np.multiply(f_filt[:,1] > 20,1)
    #Get difference of integer boolean values
    fpDiff = np.diff(fpMask)
    #Identify foot strikes based on shift from False to True
    onInd = np.where(fpDiff > 0)[0] + 1
    
    #Identify which foot is in contact based on which foot marker is closer
    #Get the right and left MT5 markers
    RMT5 = c3dAcq.GetPoint('R.MT5').GetValues()
    LMT5 = c3dAcq.GetPoint('L.MT5').GetValues()
    #Check for any gaps in markers and fill, while also interpolating to force sampling rate
    RMT5_interp = np.empty((len(f_filt),3))
    LMT5_interp = np.empty((len(f_filt),3))
    #Loop through axes
    for aa in range(3):            
        #RMT5              
        #Create a timestamp for the signal
        t = np.linspace(0,1/c3dAcq.GetPointFrequency()*(len(RMT5)-1),len(RMT5))
        #Remove nans from data array and corresponding times
        yi = RMT5[:,aa][~np.isnan(RMT5[:,aa])]
        xi = t[~np.isnan(RMT5[:,aa])]                
        #Interpolate data to fill gaps
        cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
        RMT5_interp[:,aa] = cubicF(np.linspace(t[0], t[-1], len(f_filt)))      
        #LMT5               
        #Create a timestamp for the signal
        t = np.linspace(0,1/c3dAcq.GetPointFrequency()*(len(LMT5)-1),len(LMT5))
        #Remove nans from data array and corresponding times
        yi = LMT5[:,aa][~np.isnan(LMT5[:,aa])]
        xi = t[~np.isnan(LMT5[:,aa])]                
        #Interpolate data to fill gaps
        cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
        LMT5_interp[:,aa] = cubicF(np.linspace(t[0], t[-1], len(f_filt)))                
    #Set list to store foot strikes in
    whichFoot = list()
    #Loop through the indexed force contact locations and determine foot
    for ff in range(len(onInd)):
        #Get the peak force index for the current foot strike
        if ff < len(onInd)-1:
            currForce = f_filt[onInd[ff]:onInd[ff+1],1]
            getInd = onInd[ff] + np.where(currForce == np.max(currForce))
        else:
            currForce = f_filt[onInd[ff]:-1,1]
            getInd = onInd[ff] + np.where(currForce == np.max(currForce))
        #Get the COP location at the current foot strike peak force
        currPos = cop_interp[getInd,:][0][0]
        #Get each marker position at the index
        currR = RMT5_interp[getInd,:][0][0]
        currL = LMT5_interp[getInd,:][0][0]
        #Calculate distance between points
        distR = np.sqrt(np.sum((currR-currPos)**2, axis=0))
        distL = np.sqrt(np.sum((currL-currPos)**2, axis=0))
        #Determine closer foot and set in list
        if distR < distL:
            whichFoot.append('right')
        elif distL < distR:
            whichFoot.append('left')
        
    #Identify times of foot strike
    #Set time variable for force data
    t = np.linspace(0,1/c3dAcq.GetAnalogFrequency()*(len(f_filt)-1),len(f_filt))
    #Set list to store in
    leftFS = list()
    rightFS = list()        
    #Loop through foot strikes and allocate
    for ff in range(len(whichFoot)):
        #Get current time based on index value
        currTime = t[onInd[ff]]
        #Append to appropriate list
        if whichFoot[ff] == 'left':
            leftFS.append(currTime)
        elif whichFoot[ff] == 'right':
            rightFS.append(currTime)
            
    return leftFS, rightFS

#Kinematics plots
def plotKinematics(subID, trialID, df_ik, leftFS, rightFS):
    
    #Inputs:
        #subID - string of subject ID 
        #trialID - string of trial ID
        #df_ik - dataframe containing IK results
        #leftFS - left foot strike times
        #rightFS - right foot strike times
    
    
    #Set variables to plot
    varPlot = [['pelvis_tilt'],['pelvis_list'],['pelvis_rotation'],
               ['hip_flexion_r','hip_flexion_l'],['hip_adduction_r','hip_adduction_l'],['hip_rotation_r','hip_rotation_l'],
               ['knee_angle_r','knee_angle_l'],['ankle_angle_r','ankle_angle_l']]
    plotNames = ['Pelvis Tilt', 'Pelvis List', 'Pelvis Rotation',
                 'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
                 'Knee Angle', 'Ankle Angle']
    
    #Set up variable for axes to plot on
    whichAx = [[0,0], [0,1], [0,2],
               [1,0], [1,1], [1,2],
               [2,0], [2,1]] 

    #Initialise subplot
    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (8,8))
    
    #Set overall figure title
    fig.suptitle(subID+' '+trialID+' Kinematics / Blue = Right, Red = Left', fontsize=16)
    
    #Set axes title
    for vv in range(len(plotNames)):
        ax[whichAx[vv][0],whichAx[vv][1]].set_title(plotNames[vv])
    
    #Tight layout
    plt.tight_layout()
    
    #Blank out the last unnecessary axes
    ax[2,2].set_axis_off()
    
    #Loop through angles
    for vv in range(len(plotNames)):
        
        #Check if a single or dual variable
        if len(varPlot[vv]) > 1:
        
            #Do right side first
            
            #Extract data for current variable and gait cycles
            for gc in range(len(rightFS) - 1):
                
                #Extract data between current heel strikes
                
                #Get start and end time                   
                startTime = rightFS[gc]
                endTime = rightFS[gc+1]
                
                #Create a boolean mask for in between event times
                extractTime = ((df_ik['time'] > startTime) & (df_ik['time'] < endTime)).values
                
                #Extract the time values
                timeVals = df_ik['time'][extractTime].values
                
                #Extract the data
                dataVals = df_ik[varPlot[vv][0]][extractTime].values
                
                #Normalise data to 0-100%
                newTime = np.linspace(timeVals[0],timeVals[-1],101)
                interpData = np.interp(newTime,timeVals,dataVals)
                
                #Plot interpolated data
                ax[whichAx[vv][0],whichAx[vv][1]].plot(np.linspace(0,100,101),
                                                       interpData,
                                                       color = 'blue',
                                                       linewidth = 0.5)
            #Do left side next
            
            #Extract data for current variable and gait cycles
            for gc in range(len(leftFS) - 1):
                
                #Extract data between current heel strikes
                
                #Get start and end time                   
                startTime = leftFS[gc]
                endTime = leftFS[gc+1]
                
                #Create a boolean mask for in between event times
                extractTime = ((df_ik['time'] > startTime) & (df_ik['time'] < endTime)).values
                
                #Extract the time values
                timeVals = df_ik['time'][extractTime].values
                
                #Extract the data
                dataVals = df_ik[varPlot[vv][1]][extractTime].values
                
                #Normalise data to 0-100%
                newTime = np.linspace(timeVals[0],timeVals[-1],101)
                interpData = np.interp(newTime,timeVals,dataVals)
                
                #Plot interpolated data
                ax[whichAx[vv][0],whichAx[vv][1]].plot(np.linspace(0,100,101),
                                                       interpData,
                                                       color = 'red',
                                                       linewidth = 0.5)
    
            #Set x-axes limits
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([0,100])
            
        else:
            
            #Extract data for current variable and gait cycles
            for gc in range(len(rightFS) - 1):
                
                #Extract data between current heel strikes
                
                #Get start and end time
                startTime = rightFS[gc]
                endTime = rightFS[gc+1]
                
                #Create a boolean mask for in between event times
                extractTime = ((df_ik['time'] > startTime) & (df_ik['time'] < endTime)).values
                
                #Extract the time values
                timeVals = df_ik['time'][extractTime].values
                
                #Extract the data
                dataVals = df_ik[varPlot[vv][0]][extractTime].values
                
                #Normalise data to 0-100%
                newTime = np.linspace(timeVals[0],timeVals[-1],101)
                interpData = np.interp(newTime,timeVals,dataVals)
                
                #Plot interpolated data
                ax[whichAx[vv][0],whichAx[vv][1]].plot(np.linspace(0,100,101),
                                                       interpData,
                                                       color = 'blue',
                                                       linewidth = 0.5)
        #Set x-axes limits
        ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([0,100])
        
#Save dictionary with pickle and bz2
def savePickleDict(dictObj, fullFile):
    with bz2.BZ2File(fullFile, 'w') as fileToWrite: 
        cPickle.dump(dictObj, fileToWrite)
        
#Load dictionary with pickle and bz2
def loadPickleDict(fullFile):
    fileToRead = bz2.BZ2File(fullFile, 'rb')
    fileToRead = cPickle.load(fileToRead)
    return fileToRead

#Max Cohen's d for 1D comparisons
def calcCohensD_1D(d1, d2):
    #Calculate pooled SD of two datasets
    pooledSD = np.sqrt((np.std(d1, axis = 0)**2 + 
                        np.std(d2, axis = 0)**2) / 2)
    #Calculate Cohen's D
    cohensD = (np.mean(d1, axis = 0) - 
               np.mean(d2, axis = 0)) / pooledSD
    #Calculate maximum Cohen's D and return
    maxEffect = np.max(cohensD)
    return maxEffect

#Cohen's d for 0D comparisons
def calcCohensD_0D(d1, d2):
    #Calculate pooled SD of two datasets
    pooledSD = np.sqrt((np.std(d1)**2 + 
                        np.std(d2)**2) / 2)
    #Calculate and return Cohen's D
    cohensD = (np.mean(d1) - 
               np.mean(d2)) / pooledSD
    return cohensD

# %% Set-up

#Create main directory variable
mainDir = os.getcwd()

#Create path to analysis directories
#Storage directory
os.chdir('..\\Analysis\\DataStorage')
storeAnalysisDir = os.getcwd()
#Sequential analysis directory
os.chdir('..\\SequentialAnalysis')
seqAnalysisDir = os.getcwd()

#Create path to data storage folder
os.chdir('..\\..\\Fukuchi2017-RunningDataset')
dataDir = os.getcwd()

#Get directory list
dList = os.listdir()
#Just keep directories with 'RBDS' for subject list
subList = []
for item in dList:
    if os.path.isdir(item):
        #Check for subject directory and append if appropriate
        if item.startswith('RBDS'):
            subList.append(item)
            
#Get participant info
df_subInfo = pd.read_csv('ParticipantInfo\\RBDSinfo.csv')

#Set the alpha level for the statistical tests
alpha = 0.05

#Set analysis variable
analysisVar = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
               'knee_angle_r', 'ankle_angle_r']

#Set labels for analysis variables
analysisLabels = ['Hip Flexion', 'Hip Adduction', 'Hip Rotation',
                  'Knee Flexion', 'Ankle Dorsi/Plantarflexion']

#Set colour palette for analysis variables
analysisCol = sns.color_palette('colorblind', len(analysisVar))

#Set symbols for analysis variables
analysisSym = ['s', 'o', 'd', '^', 'v']

# %% Collate processed data

#Set trial names to extract
#Each participant at least has the 25, 35 and 45 runs (some have 30)
trialList = ['runT25','runT35','runT45']

#Setting for whether to import raw data or load the pickle file
#CHANGE THIS TO FALSE IF THE PICKLE (.PBZ2) FILES DON'T EXIST
loadRawData = False

#Check for loading data
if loadRawData:
    
    #Set dictionary to store data in
    dataDict = {'subID': [], 'trialID': [], 'kinematics': [],
                'rightFS': [], 'leftFS': [], 
                'rGC_n': [], 'lGC_n': []}

    #Loop through subjects
    for ii in range(len(subList)):
    
        #Navigate to subject directory
        os.chdir(subList[ii])
        
        #Loop through trials
        for tt in range(len(trialList)):
            
            #Append subject and trial ID to data dictionary
            dataDict['subID'].append(subList[ii])
            dataDict['trialID'].append(trialList[tt])
            
            #Load in current kinematic data
            ikSto = osim.Storage(subList[ii]+trialList[tt]+'_ik.mot')
            
            #Smooth kinematic data with 4th order 10Hz FIR filter
            ikSto.lowpassFIR(4, 10)
            
            #Convert kinematics storage to dataframe
            df_ik = stoToDF(ikSto)
            
            #Append kinematics to data dictionary
            dataDict['kinematics'].append(df_ik)
            
            #Identify foot strikes from c3d data
            leftFS, rightFS = footStrikeID(subList[ii]+trialList[tt]+'.c3d')
            
            #Append to overall data dictionary
            dataDict['leftFS'].append(leftFS)
            dataDict['rightFS'].append(rightFS)
                    
            #Determine number of gait cycles for each foot and append to dictionary
            dataDict['lGC_n'].append(len(leftFS) - 1)
            dataDict['rGC_n'].append(len(rightFS) - 1)
            
            # #Plot data for later error checking
            # plotKinematics(subList[ii], trialList[tt], df_ik, leftFS, rightFS)
            
            # #Save figure
            # plt.savefig(subList[ii]+'_'+trialList[tt]+'_kinematics.png')
            # plt.close()
            
        #Print confirmation
        print('Data extracted for '+subList[ii])
            
        #Return to data directory
        os.chdir(dataDir)
        
    #Stash the compiled data for speedier use later
    savePickleDict(dataDict, storeAnalysisDir+'\\Fukuchi2017-Running-dataDict.pbz2')
    
else:
    
    #Load the already imported data dictionary
    dataDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-dataDict.pbz2')
    
    #Print confirmation
    print('Pre-loaded data imported for all participants.')
    
#Determine minimum gait cycles
#This analysis will focus on right limb data, so we'll focus on these gait cycle n's
minGC = min(dataDict['rGC_n'])

### NOTE
    #The above calculation finds 34 to be the minimum number of gait cycles across all all participants
    #Considering this we can probably safely go up to 15 and have minimum overlap
    #We can increase this to minimum when not comparing cycles within a participant
    
# %% Extract data and run tests

#Extract each participants gait cycles into time normalised arrays

#Setting for whether to normalise data or load the pickle file
#CHANGE THIS TO FALSE IF THE PICKLE (.PBZ2) FILES DON'T EXIST
normaliseData = False

#Check for normalising data
if normaliseData:
    
    #Set dictionary to store data in
    normDataDict = {key: {} for key in trialList}
    for tt in range(len(trialList)):
        normDataDict[trialList[tt]] = {key: [] for key in analysisVar}
        
    #Loop through subjects
    for ii in range(len(subList)):
        
        #Loop through analysis variables
        for vv in range(len(analysisVar)):
        
            #Loop through trial types
            for tt in range(len(trialList)):
            
                #Extract the current participants kinematic data relevant to
                #current trial tupe. Get the index corresponding to this in the
                #data dictionary.
                subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                currInd = list(set(subInd) & set(trialInd))[0]
                
                #Get the right foot strike indices
                rightFS = dataDict['rightFS'][currInd]
                
                #Set a space to store normalised data into for calculating
                #current subjects mean
                normData = np.zeros((len(rightFS)-1,101))
                
                #Get the IK dataframe
                df_ik = dataDict['kinematics'][currInd]
                
                #Loop through number of gait cycles and normalise kinematics
                for nn in range(len(rightFS)-1):
                    
                    #Extract data between current heel strikes
                
                    #Get start and end time
                    startTime = rightFS[nn]
                    endTime = rightFS[nn+1]
                    
                    #Create a boolean mask for in between event times
                    extractTime = ((df_ik['time'] > startTime) & (df_ik['time'] < endTime)).values
                    
                    #Extract the time values
                    timeVals = df_ik['time'][extractTime].values
                    
                    #Extract the data
                    dataVals = df_ik[analysisVar[vv]][extractTime].values
                    
                    #Normalise data to 0-100%
                    newTime = np.linspace(timeVals[0],timeVals[-1],101)
                    interpData = np.interp(newTime,timeVals,dataVals)                    
                    
                    #Store interpolated data in array
                    normData[nn,:] = interpData
                    
                #Append normalised data into appropriate dictionary key
                normDataDict[trialList[tt]][analysisVar[vv]].append(normData)
                
                #Print confirmation
                print(analysisVar[vv]+' data normalised and extracted for '+subList[ii]+
                      ' for '+trialList[tt]+'.')
            
    #Stash the compiled normalised data for speedier use later
    savePickleDict(normDataDict, storeAnalysisDir+'\\Fukuchi2017-Running-normDataDict.pbz2')
    
else:
    
    #Load the pre-normalised data
    normDataDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-normDataDict.pbz2')
    
    #Print confirmation
    print('Pre-loaded normalised data imported for all participants.')

# %% Determine gait cycle sample points

#Set the list of gait cycles to extract
minExtract = 5
maxExtractSingle = 30
maxExtractDual = 15
extractNo = np.linspace(minExtract,maxExtractSingle,
                        maxExtractSingle-minExtract+1)
extractNoDual = np.linspace(minExtract,maxExtractDual,
                            maxExtractDual-minExtract+1)

#Set the number of sampling iterations to run
# nSamples = 1000
nSamples = 100      ### REDUCE DURING CODE TESTING

#Setting for whether to sample gait cycles or load the pickle file
#CHANGE THIS TO FALSE IF THE PICKLE (.PBZ2) FILES DON'T EXIST
sampleGaitCycles = True #False

#Check for normalising data
if sampleGaitCycles:

    #Set dictionary to store gait cycle points in
    sampleDict = {'subID': [], 'trialID': [], 'extractNo': [],
                  'footStrikes': [], 'footStrikes1': [], 'footStrikes2': []}
    
    #Loop through subjects
    for ii in range(len(subList)):
        
        #Loop through trial list
        for tt in range(len(trialList)):
            
            #Get the index for the current subject/trial combo
            subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
            trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
            currInd = list(set(subInd) & set(trialInd))[0]
        
            #Get the number of gait cycles for this participant
            #Note that this is for the right limb this is what we'll analyse
            nGC = dataDict['rGC_n'][currInd]
            
            #Loop through extraction numbers
            for ee in range(len(extractNo)):
                
                #Determine the 'go zone' for random selection of the singular set of
                #foot strikes
                gcStarts = np.linspace(0,nGC,nGC+1)
                #Remove the final X values relative to current extraction number, as
                #these won't leave enough to grab
                goZoneSingle = gcStarts[0:int(-extractNo[ee])]
                
                #Check if this extraction number is within the boundaries for the 
                #dual comparison
                if extractNo[ee] <= maxExtractDual:
                
                    #Determine the 'go zones' for random selection of the first set
                    #Start by creating each individual cycle in a list
                    gcStarts = np.linspace(0,nGC,nGC+1)
                    #Remove the final X values relative to current extraction number, as
                    #these won't leave enough to grab
                    gcStarts = gcStarts[0:int(-extractNo[ee])]
                    #Loop through the start numbers and check if using it there will be
                    #a valid number before or after it
                    goZoneDual = list()
                    for gg in range(len(gcStarts)):
                        #Check whether there will be valid values after
                        enoughAfter = gcStarts[-1] - (gcStarts[gg] + extractNo[ee]) > 0
                        #Check whether there would be valid values before
                        enoughBefore = gcStarts[gg] - extractNo[ee] > 0
                        #If one of these is True then the value can be added to the 'go zone'
                        if enoughAfter or enoughBefore:
                            goZoneDual.append(gcStarts[gg])
                
                #Set seed here for sampling consistency
                random.seed(12345)
                
                #Loop through sampling number
                for ss in range(nSamples):
                    
                    #Append subject and trial ID names
                    sampleDict['subID'].append(subList[ii])
                    sampleDict['trialID'].append(trialList[tt])
                    
                    #Append extract number details
                    sampleDict['extractNo'].append(int(extractNo[ee]))
                    
                    #Select a random number from the single 'go zone'
                    singlePick = random.choice(goZoneSingle)
                    
                    #Set strikes for current sample in directory
                    sampleDict['footStrikes'].append(list(map(round,list(np.linspace(singlePick,singlePick+int(extractNo[ee]),int(extractNo[ee]+1))))))
                    
                    #Check if dual selection is necessary
                    if extractNo[ee] <= maxExtractDual:
                    
                        #Select a random number from the list to start from
                        dualPick1 = random.choice(goZoneDual)
                        
                        #Set a list to make second selection from
                        select2 = list()
                        
                        #At this point split into two lists so length checks are easier
                        #Can't use preceding starting points if they are within the
                        #extraction number of the starting point
                        goZone1 = [x for x in goZoneDual if x < dualPick1-extractNo[ee]+1]
                        #Can't use values that will be encompassed within the gait cycles
                        #extracted from the first starting point
                        goZone2 = [x for x in goZoneDual if x > dualPick1+extractNo[ee]-1]
                        #Concatenate the lists for to select from
                        select2 = goZone1 + goZone2
        
                        #Select a random number from the second list to start from
                        dualPick2 = random.choice(select2)
                    
                        #Set strikes for current sample in dictionary
                        sampleDict['footStrikes1'].append(list(map(round,list(np.linspace(dualPick1,dualPick1+int(extractNo[ee]),int(extractNo[ee]+1))))))
                        sampleDict['footStrikes2'].append(list(map(round,list(np.linspace(dualPick2,dualPick2+int(extractNo[ee]),int(extractNo[ee]+1))))))
                    
                    #Otherwise just set nan's
                    else:
                        sampleDict['footStrikes1'].append(np.nan)
                        sampleDict['footStrikes2'].append(np.nan)
        
        #Print confirmation for subject
        print('Cycle sample numbers extracted for '+subList[ii])
    
    #Stash the compiled sampling vals for speedier use later
    savePickleDict(sampleDict, storeAnalysisDir+'\\Fukuchi2017-Running-sampleDict.pbz2')
    
else:
    
    #Load the pre-sampled gait cycle data
    sampleDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-sampleDict.pbz2')

    #Print confirmation
    print('Pre-sampled gait cycle numbers imported.')

#Convert dictionary to a dataframe
df_samples = pd.DataFrame.from_dict(sampleDict)
    
# %% Data analysis

# The below section undertakes a series of analyses to answer the specific research
# questions (RQs) for this study.Below cell sections include specific notes on
# the various RQs prior to them being encapsulated within a set series of loops.
# The below analysis and sampling procedures are time intensive, hence the pre
# analysed data is available and can be loaded in by keeping the below variable
# set to False. If analysis needs to be undertaken, set this to True.
analyseData = True #False

# %% RQ1: Sequential analysis with increasing gait cycle number

##### TODO: consider appropriateness of this vs. sequential analysis of a series
##### of trials --- i.e. we take a random sample of 5 gait cycles from an individuals
##### data - how often do they reach stability?

# This analysis follows a similar line to:
    #
    # Taylor et al. (2015). Determining optimal trial size using sequential
    # analysis. J Sports Sci, 33: 300-308
    #
    # Forrester (2015). Selecting the number of trials in experimental
    # biomechanics studies. Int Biomech, 2: 62-72.
#
# Each discrete point of the time-normalised gait cycle is considered its own
# entity, and a moving point mean is calculated as gait cycle number increases
# across the trial. This moving point mean is compared to the mean and 0.25 SD.
# Each point will fit within this bandwidth differently, so we can look at every
# point - but for true 'stability' you probably want every point to fall within
# this bandwidth, right?
#
# Here we also consider 0D peak variables as a comparison to the 1D curves.

# %% RQ2: Varying cycle number comparison to 'ground truth' mean
    
# This analysis compares a sample of an individuals treadmill bout to their 
# entire treadmill bout (i.e. 'ground truth'). Given we don't know exactly what
# an individuals 'true' kinematics are, we take the entire treadmill bout as this
# 'truth' value --- and this is supported by the sequential analysis results
# where it appears most (if not all) have a pretty stable mean. 
#
# The sampled mean (with varying number of gait cycles) is compared to our 'ground
# truth' mean, and given these come from the same treadmill running bout we effectively
# want to see no statistical difference. With the alpha level set at 0.05, we do
# however consider a false-positive rate of 5% to come up.
#
# Here we consider 0D peak variables as week as the 1D curves.

# %% RQ3: Comparing samples from different sections of trial

##### TODO: this is repetitive of the above sample cycling, so perhaps encase
##### both within same loop to speed up process...

# %% RQ4: Cycle sampling and number effect on refuting null hypothesis

# This section examines the effect of gait cycle number and sampling on the findings
# from null hypothesis testing, specifically the effect of speed on gait biomechanics.
# The original work of Fukuchi et al. (2017) examined this question, and found 
# a series of biomechanical variables are impacted by gait speed. Here we iteratively
# replicate this hypothesis testing with varying numbers and differently sampled
# gait cycles contributing to participant data. The theory being tested here is
# how does the number and selection of gait cycles impact the answers to our hypothesis
# testing.

##### TODO: considerations around 0D vs. 1D variables, ANOVA vs. t-test

##### To start with, replicate the Fukuchi et al. analysis...

# %% Start the analysis...

##### TODO: from a data storage/file size perspective, do we really need the
##### extracted data from every sample iteration to be stored??? Even in a compressed
##### state this produces massive files (e.g. the groundTruthDict is 624MB even
##### on the 100 sample iterations!)
#####
##### It's potentially necessary to actually probe certain findings, and may be
##### just a by-product we have to cop with what we're doing...

#Set a dictionary to store findings of sequential analysis (RQ1)
seqDict = {'nGC': [], 'subID': [], 'trialID': [], 'analysisVar': [],
           'seqVal': [], 'varType': []}

#Set a dictionary to store the ground truth comparisons to (RQ2)
groundTruthDict = {'extractNo': [], 'trialID': [], 'varType': [],
                   'rejectH0': [], 'pVal': [],
                   'analysisVar': [],
                   'groundTruth': [], 'extract': [],
                   'groundTruth_m': [], 'extract_m': [],
                   'meanAbsError': [], 'effectSize': []}

#Set a dictionary to store findings of comparing different samples of gait cycles (RQ3)
compDict = {'extractNo': [], 'trialID': [], 'varType': [],
            'rejectH0': [], 'pVal': [],
            'analysisVar': [],
            'Y1': [], 'Y2': [],
            'Y1_m': [], 'Y2_m': [],
            'meanAbsError': [], 'effectSize': []}

#Set dictionary to store extracted values in for 0D ANOVA (RQ4)
anovaDataDict = {'sampleNo': [], 'extractNo': [],
                 'vals': [], 'varType': [], 
                 'speed': []}

# #Set dictionary to store 0D ANOVA results (RQ4)
# anovaDict = {'extractNo': [], 'analysisVar': [], 'startPoint': [],
#              'analysisData': [], 'mean': [], 'sd': [],
#              'aovrmResults': [], 'F': [], 'p': [], 'rejectH0': []}

# #Set dictionary to store 0D post-hoc pairwise results (RQ4)
# pairwiseDict = {'extractNo': [], 'analysisVar': [], 'startPoint': [],
#                 'comparison': [], 'val0D': [], 'mean': [], 'sd': [],
#                 't': [], 'p': [], 'rejectH0': []}

#Calculate total iterations for progressive timing and time left estimates
totalIter = len(trialList) * len(analysisVar) * len(extractNo)

#Set a current interations variable
currIter = 0

#Start timer
t0 = time.time()

#Loop through trial types
for tt in range(len(trialList)):
    
    #Extract the dataframe for the current trial ID
    df_currTrial = df_samples.loc[df_samples['trialID'] == trialList[tt],]
    
    #Loop through variables
    for vv in range(len(analysisVar)):
        
        #Calculate the 'ground truth' values based on all of the participants
        #gait cycles for the current variables        
        #Set the 'ground truth' mean array
        groundTruth_0D = np.zeros((len(subList),1))
        groundTruth_1D = np.zeros((len(subList),101))
        
        #Loop through participants
        for ii in range(len(subList)):

            #Extract the current participants kinematic data relevant to
            #current trial type. Get the index corresponding to this in the
            #data dictionary.
            subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
            trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
            currInd = list(set(subInd) & set(trialInd))[0]
            
            #Get the right foot strike indices
            rightFS = dataDict['rightFS'][currInd]
            
            #Extract participants normalised data for the current trial and variable
            normDataAll = normDataDict[trialList[tt]][analysisVar[vv]][ii]
                
            #Calculate the mean/SD of all of the current subjects normalised data
            total_m = np.mean(normDataAll, axis = 0)
            total_sd = np.std(normDataAll, axis = 0)
            
            #Calculate the peak 0D variable mean/SD of all normalised data
            peak_m = np.mean(np.max(normDataAll, axis = 1), axis = 0)
            peak_sd = np.std(np.max(normDataAll, axis = 1), axis = 0)
            
            #Calculate the mean of the current subjects normalised data
            #Store in the current ground truths array for SPM1D analysis
            #Also calculate the mean of the peaks here and store in 0D array
            groundTruth_1D[ii,:] = np.mean(normDataAll, axis = 0)
            groundTruth_0D[ii,0] = np.mean(np.max(normDataAll, axis = 1), axis = 0)
            
            #Loop through n+1 number of gait cycles sequentially and assess
            #points relative to +/- 0.25 SD bounds
            for nn in range(1,len(rightFS)-1):
                
                #Calculate mean and SD for current number of cycles
                curr_m = np.mean(normDataAll[0:nn+1,:], axis = 0)
                
                #Normalise to zero mean and 1 SD
                curr_norm = (curr_m - total_m) / total_sd
                
                #Calculate peak mean for 0D variable
                curr_peak_m = np.mean(np.max(normDataAll[0:nn+1,:], axis = 1), axis = 0)
                
                #Normalise to zero mean and 1SD
                curr_peak_norm = (curr_peak_m - peak_m) / peak_sd
                
                #Add to data dictionary
                #Include calculations for mean, min and max sequential variables
                
                #1D values
                seqDict['nGC'].append(nn+1)
                seqDict['subID'].append(subList[ii])
                seqDict['trialID'].append(trialList[tt])
                seqDict['analysisVar'].append(analysisVar[vv])
                seqDict['seqVal'].append(np.max(np.abs(curr_norm)))
                seqDict['varType'].append('1D')
                
                #0D values
                seqDict['nGC'].append(nn+1)
                seqDict['subID'].append(subList[ii])
                seqDict['trialID'].append(trialList[tt])
                seqDict['analysisVar'].append(analysisVar[vv])
                seqDict['seqVal'].append(curr_peak_norm)
                seqDict['varType'].append('0D')
            
        #Flatten 0D ground truth array
        groundTruth_0D = groundTruth_0D.flatten()
        
        #Loop through the extraction numbers
        for ee in range(len(extractNo)):
            
            #Set current extract number
            currNo = int(extractNo[ee])
            
            #Extract the dataframe for the current extraction number
            df_currExtract = df_currTrial.loc[df_currTrial['extractNo'] == currNo,]
            
            #Loop through the sampling number
            for ss in tqdm.tqdm(range(nSamples),
                                desc = 'Sampling '+str(currNo)+' gait cycles for '+
                                analysisVar[vv]+' from '+trialList[tt]):
            # for ss in range(nSamples):
                
                #Set array to store the extracted datasets for this sample iteration
                #that will be compared to the ground truth dataset
                #Both 0D and 1D variables
                extract_1D = np.zeros((len(subList),101))
                extract_0D = np.zeros((len(subList),1))
                
                #Set array to store the extracted datasets for this sample iteration
                #that will be compared to one another
                #Both 0D and 1D variables
                #Not that this analysis is only undertaken if the current gait
                #cycle extraction number is relevant for comparing the two
                if currNo <= max(extractNoDual):
                    Y1_1D = np.zeros((len(subList),101))
                    Y2_1D = np.zeros((len(subList),101))
                    Y1_0D = np.zeros((len(subList),1))
                    Y2_0D = np.zeros((len(subList),1))
                
                #Loop through subjects and get their data for the current sample
                for ii in range(len(subList)):
                                        
                    #Extract the dataframe that matches the current subject
                    #As part of this also extract the row that matches the current
                    #sampling iteration index
                    df_currSub = df_currExtract.loc[df_currExtract['subID'] == subList[ii],]
                    df_currIter = df_currSub.iloc[ss]
                    
                    #Extract the current participants kinematic data relevant to
                    #current trial tupe. Get the index corresponding to this in the
                    #data dictionary.
                    subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                    trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                    currInd = list(set(subInd) & set(trialInd))[0]
                    
                    #Get the right foot strike indices
                    rightFS = dataDict['rightFS'][currInd]
                    
                    #Extract participants normalised data for the current trial and variable
                    normDataAll = normDataDict[trialList[tt]][analysisVar[vv]][ii]
                    
                    #Set a space to store normalised data for current extraction
                    #for the singular set of foot strikes
                    normDataSingle = np.zeros((currNo,101))
                    
                    #Set a space to store normalised data for current extraction
                    #for the comparative set of foot strikes
                    #Again, this is only relevant if the current gait cycle extraction
                    #number sits within the dual gait cycle extraction values
                    if currNo <= max(extractNoDual):
                        normDataDual1 = np.zeros((currNo,101))
                        normDataDual2 = np.zeros((currNo,101))
                    
                    #Loop through extraction number and get normalised kinematics
                    for nn in range(currNo):
                        
                        #Grab the appropriate gait cycle and append to array
                        normDataSingle[nn,:] = normDataAll[df_currIter['footStrikes'][nn]]
                        if currNo <= max(extractNoDual):
                            normDataDual1[nn,:] = normDataAll[df_currIter['footStrikes1'][nn]]
                            normDataDual2[nn,:] = normDataAll[df_currIter['footStrikes2'][nn]]
                
                    #Calculate the mean of the current subjects normalised data
                    #Store in the current sample iterations array for SPM1D analysis
                    #For the singular gait cycle set data
                    extract_1D[ii,:] = np.mean(normDataSingle, axis = 0)
                    extract_0D[ii,0] = np.mean(np.max(normDataSingle, axis = 1), axis = 0)
                    #For the comparative gait cycle set data
                    if currNo <= max(extractNoDual):
                        Y1_1D[ii,:] = np.mean(normDataDual1, axis = 0)
                        Y1_0D[ii,0] = np.mean(np.max(normDataDual1, axis = 1), axis = 0)
                        Y2_1D[ii,:] = np.mean(normDataDual2, axis = 0)
                        Y2_0D[ii,0] = np.mean(np.max(normDataDual2, axis = 1), axis = 0)
                    
                #Flatten 0D arrays
                extract_0D = extract_0D.flatten()
                if currNo <= max(extractNoDual):
                    Y1_0D = Y1_0D.flatten()
                    Y2_0D = Y2_0D.flatten()
                
                #RQ2 statistical analysis
                
                #Conduct the SPM1D t-test on comparing ground truth to extracted
                t_RQ2 = spm1d.stats.ttest_paired(groundTruth_1D, extract_1D)
                ti_RQ2 = t_RQ2.inference(alpha, two_tailed = True, interp = True)
                
                # #Visualise ground truth vs. extracted comparison
                # #Set-up plot
                # plt.figure(figsize=(8, 3.5))
                # #Plot mean and SD of two samples
                # ax1 = plt.axes((0.1, 0.15, 0.35, 0.8))
                # spm1d.plot.plot_mean_sd(groundTruth_1D, linecolor = 'b', facecolor = 'b')
                # spm1d.plot.plot_mean_sd(extract_1D, linecolor = 'r', facecolor='r')
                # ax1.axhline(y = 0, color = 'k', linestyle=':')
                # ax1.set_xlabel('0-100% Gait Cycle')
                # ax1.set_ylabel(analysisVar[vv])
                # #Plot SPM results
                # ax2 = plt.axes((0.55,0.15,0.35,0.8))
                # ti_RQ2.plot()
                # ti_RQ2.plot_threshold_label(fontsize = 8)
                # ti_RQ2.plot_p_values(size = 10, offsets = [(0,0.3)])
                # ax2.set_xlabel('0-100% Gait Cycle')
                # #Show plot
                # plt.show()
                
                #Calculate mean absolute error and effect between current two curves
                groundTruth_1D_m = np.mean(groundTruth_1D, axis = 0)
                extract_1D_m = np.mean(extract_1D, axis = 0)
                mae_1D_RQ2 = np.mean(abs(groundTruth_1D_m - extract_1D_m))
                cohens_1D_RQ2 = calcCohensD_1D(groundTruth_1D, extract_1D)
                
                #Conduct paired t-test on peak values
                t0_RQ2,p0_RQ2 = stats.ttest_rel(groundTruth_0D, extract_0D)
                if p0_RQ2 < 0.05:
                    h0reject_0D_RQ2 = True
                else:
                    h0reject_0D_RQ2 = False
                    
                #Calculate mean absolute error and effect between current 0D variables
                groundTruth_0D_m = np.mean(groundTruth_0D, axis = 0)
                extract_0D_m = np.mean(extract_0D, axis = 0)
                mae_0D_RQ2 = np.mean(abs(groundTruth_0D_m - extract_0D_m))
                cohens_0D_RQ2 = calcCohensD_0D(groundTruth_0D, extract_0D)
                
                #RQ3 statistical analysis
                if currNo <= max(extractNoDual):
                
                    #Conduct the SPM1D t-test comparing the two extracted samples
                    t_RQ3 = spm1d.stats.ttest_paired(Y1_1D, Y2_1D)
                    ti_RQ3 = t_RQ3.inference(alpha, two_tailed = True, interp = True)
                    
                    # #Visualise two extracted samples comparison
                    # #Set-up plot
                    # plt.figure(figsize=(8, 3.5))
                    # #Plot mean and SD of two samples
                    # ax1 = plt.axes((0.1, 0.15, 0.35, 0.8))
                    # spm1d.plot.plot_mean_sd(Y1_1D, linecolor = 'b', facecolor = 'b')
                    # spm1d.plot.plot_mean_sd(Y1_1D, linecolor = 'r', facecolor='r')
                    # ax1.axhline(y = 0, color = 'k', linestyle=':')
                    # ax1.set_xlabel('0-100% Gait Cycle')
                    # ax1.set_ylabel(analysisVar[vv])
                    # #Plot SPM results
                    # ax2 = plt.axes((0.55,0.15,0.35,0.8))
                    # ti_RQ3.plot()
                    # ti_RQ3.plot_threshold_label(fontsize = 8)
                    # ti_RQ3.plot_p_values(size = 10, offsets = [(0,0.3)])
                    # ax2.set_xlabel('0-100% Gait Cycle')
                    # #Show plot
                    # plt.show()
                    
                    #Calculate mean absolute error and effect between two sampled curves
                    Y1_1D_m = np.mean(Y1_1D, axis = 0)
                    Y2_1D_m = np.mean(Y2_1D, axis = 0)
                    mae_1D_RQ3 = np.mean(abs(Y1_1D_m - Y2_1D_m))
                    cohens_1D_RQ3 = calcCohensD_1D(Y1_1D, Y2_1D)
                    
                    #Conduct paired t-test on peak values
                    t0_RQ3,p0_RQ3 = stats.ttest_rel(Y1_0D, Y2_0D)
                    if p0_RQ3 < 0.05:
                        h0reject_0D_RQ3 = True
                    else:
                        h0reject_0D_RQ3 = False
                        
                    #Calculate mean absolute error and effect for 0D variable between two samples
                    Y1_0D_m = np.mean(Y1_0D, axis = 0)
                    Y2_0D_m = np.mean(Y2_0D, axis = 0)
                    mae_0D_RQ3 = np.mean(abs(Y1_0D_m - Y2_0D_m))
                    cohens_0D_RQ3 = calcCohensD_0D(Y1_0D, Y2_0D)
                
                #RQ2 data appending
                
                #Collate results from this ground truth comparison into dictionary
                #1D
                groundTruthDict['extractNo'].append(currNo)
                groundTruthDict['trialID'].append(trialList[tt])
                groundTruthDict['varType'].append('1D')
                groundTruthDict['rejectH0'].append(ti_RQ2.h0reject)
                groundTruthDict['pVal'].append(ti_RQ2.p) #note there are no p-values for non-statistically significant results            
                groundTruthDict['analysisVar'].append(analysisVar[vv])
                groundTruthDict['groundTruth'].append(groundTruth_1D)
                groundTruthDict['extract'].append(extract_1D)
                groundTruthDict['groundTruth_m'].append(groundTruth_1D_m)
                groundTruthDict['extract_m'].append(extract_1D_m)
                groundTruthDict['meanAbsError'].append(mae_1D_RQ2)
                groundTruthDict['effectSize'].append(cohens_1D_RQ2)
                #0D
                groundTruthDict['extractNo'].append(currNo)
                groundTruthDict['trialID'].append(trialList[tt])
                groundTruthDict['varType'].append('0D')
                groundTruthDict['rejectH0'].append(h0reject_0D_RQ2)
                groundTruthDict['pVal'].append(p0_RQ2)                
                groundTruthDict['analysisVar'].append(analysisVar[vv])
                groundTruthDict['groundTruth'].append(groundTruth_0D)
                groundTruthDict['extract'].append(extract_0D)
                groundTruthDict['groundTruth_m'].append(groundTruth_0D_m)
                groundTruthDict['extract_m'].append(extract_0D_m)
                groundTruthDict['meanAbsError'].append(mae_0D_RQ2)
                groundTruthDict['effectSize'].append(cohens_0D_RQ2)
                
                #RQ3 data appending
                if currNo <= max(extractNoDual):
                
                    #Collate results from this two sample comparison into dictionary
                    #1D
                    compDict['extractNo'].append(currNo)
                    compDict['trialID'].append(trialList[tt])
                    compDict['varType'].append('1D')
                    compDict['rejectH0'].append(ti_RQ3.h0reject)
                    compDict['pVal'].append(ti_RQ3.p) #note there are no p-values for non-statistically significant results
                    compDict['analysisVar'].append(analysisVar[vv])
                    compDict['Y1'].append(Y1_1D)
                    compDict['Y2'].append(Y2_1D)
                    compDict['Y1_m'].append(Y1_1D_m)
                    compDict['Y2_m'].append(Y2_1D_m)
                    compDict['meanAbsError'].append(mae_1D_RQ3)
                    compDict['effectSize'].append(cohens_1D_RQ3)
                    #0D
                    compDict['extractNo'].append(currNo)
                    compDict['trialID'].append(trialList[tt])
                    compDict['varType'].append('0D')
                    compDict['rejectH0'].append(h0reject_0D_RQ3)
                    compDict['pVal'].append(p0_RQ3)                
                    compDict['analysisVar'].append(analysisVar[vv])
                    compDict['Y1'].append(Y1_0D)
                    compDict['Y2'].append(Y2_0D)
                    compDict['Y1_m'].append(Y1_0D_m)
                    compDict['Y2_m'].append(Y2_0D_m)
                    compDict['meanAbsError'].append(mae_0D_RQ3)
                    compDict['effectSize'].append(cohens_0D_RQ3)
                    
                #Store the values in dictionary for RQ4 0D ANOVA
                anovaDataDict['sampleNo'].append(ss)
                anovaDataDict['vals'].append(extract_0D)
                anovaDataDict['varType'].append('0D')
                anovaDataDict['extractNo'].append(currNo)
                anovaDataDict['speed'].append(trialList[tt])
                
                #Store the values in dictionary for RQ4 0D ANOVA
                anovaDataDict['sampleNo'].append(ss)
                anovaDataDict['vals'].append(extract_1D)
                anovaDataDict['varType'].append('1D')
                anovaDataDict['extractNo'].append(currNo)
                anovaDataDict['speed'].append(trialList[tt])
                
            #Print completion notice for the current cycle
            print('\nCompleted analysis for '+str(currNo)+' gait cycles of '+
                  analysisVar[vv]+' from '+trialList[tt])
            
            #Get the new time to compare to the starting time
            tCurr = time.time()
            
            #Add an iteration and print how far through we are
            currIter = currIter + 1
            perProg = currIter / totalIter * 100            
            print(str(round(perProg,2))+'% through analysis...')
            
            #Calculate and print the estimated time remaining (in mins)
            #Only need to display estimated time if current iteration isn't the last
            if currIter != totalIter:
                avgIterTime = (tCurr - t0) / currIter
                estRemTime = (avgIterTime * (totalIter - currIter)) / 60 / 60
                print('Estimated analysis time remaining: '+str(round(estRemTime,2))+' hours!')
                
#Stash the compiled analysis dictionaries for later use
##### TODO: these are currently on 100 sample simulations...
savePickleDict(seqDict, storeAnalysisDir+'\\Fukuchi2017-Running-seqDict.pbz2')
savePickleDict(groundTruthDict, storeAnalysisDir+'\\Fukuchi2017-Running-groundTruthDict.pbz2')
savePickleDict(compDict, storeAnalysisDir+'\\Fukuchi2017-Running-compDict.pbz2')
savePickleDict(anovaDataDict, storeAnalysisDir+'\\Fukuchi2017-Running-anovaDataDict.pbz2')

# seqDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-seqDict.pbz2')
# groundTruthDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-groundTruthDict.pbz2')
# compDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-compDict.pbz2')
# anovaDataDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-anovaDataDict.pbz2')

# %%


#Convert dictionary to a dataframe
df_seqAnalysis = pd.DataFrame.from_dict(seqDict)

#Determine the point of 'stability' across participants and variables at different
#standard deviation thresholds. The works mentioned above use a 0.25 SD threshold,
#however we can also investigate other values to provide an indication of 'stability'
#at different levels

#Set variable for stability levels
stabilityLevels = [0.25, 0.50, 0.75, 1.00]

#Set dictionary to store data in
seqResultsDict = {'subID': [], 'trialID': [], 'analysisVar': [], 'varType': [],
                  'stabilityThreshold': [], 'stabilityGC': []}

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through variables
    for vv in range(len(analysisVar)):
        
        #Loop through participants and calculate individual stability
        for ii in range(len(subList)):
            
            #Extract data for the current participant, trial and variable
            #0D variable type
            df_currSeq_0D = df_seqAnalysis.loc[(df_seqAnalysis['trialID'] == trialList[tt]) &
                                               (df_seqAnalysis['analysisVar'] == analysisVar[vv]) &
                                               (df_seqAnalysis['subID'] == subList[ii]) &
                                               (df_seqAnalysis['varType'] == '0D'),].reset_index(drop = True)
            #1D variable type
            df_currSeq_1D = df_seqAnalysis.loc[(df_seqAnalysis['trialID'] == trialList[tt]) &
                                               (df_seqAnalysis['analysisVar'] == analysisVar[vv]) &
                                               (df_seqAnalysis['subID'] == subList[ii]) &
                                               (df_seqAnalysis['varType'] == '1D'),].reset_index(drop = True)
            
            #Loop through stability thresholds and extract number of gait cycles
            #when the stability value is reached
            for ss in range(len(stabilityLevels)):
                
                #Do 0D variable first
                
                #Set start gait cycle value to start check at
                checkGC = np.min(df_currSeq_0D['nGC'].unique())
                
                #Identify max gait cycle number
                maxGC = np.max(df_currSeq_0D['nGC'].unique())
                
                #Use while loop to search through gait cycle numbers
                while checkGC < maxGC:
                    
                    #Get value for current check
                    currVal = df_currSeq_0D.loc[df_currSeq_0D['nGC'] == checkGC, 'seqVal'].values[0]
                    
                    #Check if current value is under threshold
                    if np.abs(currVal) < stabilityLevels[ss]:
                        
                        #Check if the remaining values all remain under threshold
                        
                        #Get the remaining values
                        remVals = np.abs(df_currSeq_0D.loc[df_currSeq_0D['nGC'] > checkGC, 'seqVal'].values)
                        
                        #Check if all values are under the threshold
                        if (remVals < stabilityLevels[ss]).all():
                            
                            #Set the current gait cycle check in dictionary
                            #and append other info
                            seqResultsDict['subID'].append(subList[ii])
                            seqResultsDict['trialID'].append(trialList[tt])
                            seqResultsDict['analysisVar'].append(analysisVar[vv])
                            seqResultsDict['varType'].append('0D')
                            seqResultsDict['stabilityThreshold'].append(stabilityLevels[ss])
                            seqResultsDict['stabilityGC'].append(checkGC)
                            
                            #Break loop
                            break
                        
                        else:
                            
                            #Move on to the next gait cycle value
                            checkGC = checkGC + 1
                        
                    else:
                        
                        #Move on to the next gait cycle value
                        checkGC = checkGC + 1
                
                #Do a quick check to see if the checkGC is the same as the maxGC
                #If this is the case, then the data only reach stability by the
                #last gait cycle --- effectively the same as the mean --- and 
                #won't have had data added
                if checkGC == maxGC:
                    
                    #Append a nan value for the current scenario
                    seqResultsDict['subID'].append(subList[ii])
                    seqResultsDict['trialID'].append(trialList[tt])
                    seqResultsDict['analysisVar'].append(analysisVar[vv])
                    seqResultsDict['varType'].append('0D')
                    seqResultsDict['stabilityThreshold'].append(stabilityLevels[ss])
                    seqResultsDict['stabilityGC'].append(np.nan)
                    
                #Repeat with 1D variable
                
                #Set start gait cycle value to start check at
                checkGC = np.min(df_currSeq_1D['nGC'].unique())
                
                #Identify max gait cycle number
                maxGC = np.max(df_currSeq_1D['nGC'].unique())
                
                #Use while loop to search through gait cycle numbers
                while checkGC < maxGC:
                    
                    #Get value for current check
                    currVal = df_currSeq_1D.loc[df_currSeq_1D['nGC'] == checkGC, 'seqVal'].values[0]
                    
                    #Check if current value is under threshold
                    if np.abs(currVal) < stabilityLevels[ss]:
                        
                        #Check if the remaining values all remain under threshold
                        
                        #Get the remaining values
                        remVals = np.abs(df_currSeq_1D.loc[df_currSeq_1D['nGC'] > checkGC, 'seqVal'].values)
                        
                        #Check if all values are under the threshold
                        if (remVals < stabilityLevels[ss]).all():
                            
                            #Set the current gait cycle check in dictionary
                            #and append other info
                            seqResultsDict['subID'].append(subList[ii])
                            seqResultsDict['trialID'].append(trialList[tt])
                            seqResultsDict['analysisVar'].append(analysisVar[vv])
                            seqResultsDict['varType'].append('1D')
                            seqResultsDict['stabilityThreshold'].append(stabilityLevels[ss])
                            seqResultsDict['stabilityGC'].append(checkGC)
                            
                            #Break loop
                            break
                        
                        else:
                            
                            #Move on to the next gait cycle value
                            checkGC = checkGC + 1
                        
                    else:
                        
                        #Move on to the next gait cycle value
                        checkGC = checkGC + 1
                
                #Do a quick check to see if the checkGC is the same as the maxGC
                #If this is the case, then the data only reach stability by the
                #last gait cycle --- effectively the same as the mean --- and 
                #won't have had data added
                if checkGC == maxGC:
                    
                    #Append a nan value for the current scenario
                    seqResultsDict['subID'].append(subList[ii])
                    seqResultsDict['trialID'].append(trialList[tt])
                    seqResultsDict['analysisVar'].append(analysisVar[vv])
                    seqResultsDict['varType'].append('1D')
                    seqResultsDict['stabilityThreshold'].append(stabilityLevels[ss])
                    seqResultsDict['stabilityGC'].append(np.nan)
                    
                #Print confirmation
                print('Sequential results extracted for '+subList[ii]+
                  ' for '+analysisVar[vv]+' during '+trialList[tt]+
                  ' at stability threshold of '+str(stabilityLevels[ss])+' SD.')
                    
            #Print confirmation
            print('Sequential results extracted for '+subList[ii]+
                  ' for '+analysisVar[vv]+' during '+trialList[tt])
        
        #Print confirmation
        print('Sequential results extracted for '+analysisVar[vv]+
              '. '+str(vv+1)+' of '+str(len(analysisVar))+
              ' variables completed for '+trialList[tt])
    
    #Print confirmation
    print('Sequential results extracted for '+trialList[tt]+
          '. '+str(tt+1)+' of '+str(len(trialList))+' trial types completed.')

#Stash the compiled sequential results for speedier use later
savePickleDict(seqResultsDict, storeAnalysisDir+'\\Fukuchi2017-Running-seqResultsDict.pbz2')
# seqResultsDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-seqResultsDict.pbz2')

#Convert to dataframe
df_seqResults = pd.DataFrame.from_dict(seqResultsDict)

#Calculate mean and 95% CIs for how long each variable takes to reach 'stability'
#Other summary statistics like median, min and max values are also calculated
#This is done across both 0D and 1D variables, and different run trials

#Set dictionary to store data in
seqSummaryDict = {'trialID': [], 'analysisVar': [], 'varType': [], 'stabilityThreshold': [],
                   'mean': [], 'sd': [], 'median': [], 'min': [], 'max': [], 'ci95': []}

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through variables
    for vv in range(len(analysisVar)):
        
        #Loop through stability thresholds
        for ss in range(len(stabilityLevels)):
        
            #Get the relevant values to calculate
            calcValues_0D = df_seqResults.loc[(df_seqResults['trialID'] == trialList[tt]) &
                                              (df_seqResults['analysisVar'] == analysisVar[vv]) &
                                              (df_seqResults['varType'] == '0D') &
                                              (df_seqResults['stabilityThreshold'] == stabilityLevels[ss]),
                                              ['stabilityGC']]['stabilityGC'].values
            calcValues_1D = df_seqResults.loc[(df_seqResults['trialID'] == trialList[tt]) &
                                              (df_seqResults['analysisVar'] == analysisVar[vv]) &
                                              (df_seqResults['varType'] == '1D') &
                                              (df_seqResults['stabilityThreshold'] == stabilityLevels[ss]),
                                              ['stabilityGC']]['stabilityGC'].values
            
            #Calculate mean, median, min and max values
            #Mean and SD
            Xbar_0D = np.mean(calcValues_0D)
            Xbar_1D = np.mean(calcValues_1D)
            Xsd_0D = np.std(calcValues_0D)
            Xsd_1D = np.std(calcValues_1D)
            #Median
            Xtilde_0D = int(np.ceil(np.median(calcValues_0D)))
            Xtilde_1D = int(np.ceil(np.median(calcValues_1D)))
            #Min and max
            Xmin_0D = np.min(calcValues_0D)
            Xmax_0D = np.max(calcValues_0D)
            Xmin_1D = np.min(calcValues_1D)
            Xmax_1D = np.max(calcValues_1D)
            
            #Calculate 95% CI val
            ci95_0D = 1.96 * (Xsd_0D / np.sqrt(len(calcValues_0D)))
            ci95_1D = 1.96 * (Xsd_1D / np.sqrt(len(calcValues_1D)))
            
            #Append to dictionary
            #0D
            seqSummaryDict['trialID'].append(trialList[tt])
            seqSummaryDict['analysisVar'].append(analysisVar[vv])
            seqSummaryDict['varType'].append('0D')
            seqSummaryDict['stabilityThreshold'].append(stabilityLevels[ss])
            seqSummaryDict['mean'].append(Xbar_0D)
            seqSummaryDict['sd'].append(Xsd_0D)
            seqSummaryDict['median'].append(Xtilde_0D)
            seqSummaryDict['min'].append(Xmin_0D)
            seqSummaryDict['max'].append(Xmax_0D)
            seqSummaryDict['ci95'].append(ci95_0D)
            #1D
            seqSummaryDict['trialID'].append(trialList[tt])
            seqSummaryDict['analysisVar'].append(analysisVar[vv])
            seqSummaryDict['varType'].append('1D')
            seqSummaryDict['stabilityThreshold'].append(stabilityLevels[ss])
            seqSummaryDict['mean'].append(Xbar_1D)
            seqSummaryDict['sd'].append(Xsd_1D)
            seqSummaryDict['median'].append(Xtilde_1D)
            seqSummaryDict['min'].append(Xmin_1D)
            seqSummaryDict['max'].append(Xmax_1D)
            seqSummaryDict['ci95'].append(ci95_1D)

#Stash the compiled sequential summary for speedier use later
savePickleDict(seqSummaryDict, storeAnalysisDir+'\\Fukuchi2017-Running-seqSummaryDict.pbz2')
# seqSummaryDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-seqSummaryDict.pbz2')

#Convert to dataframe
df_seqSummary = pd.DataFrame.from_dict(seqSummaryDict)

#Export sequential analysis results to file
df_seqSummary.to_csv(seqAnalysisDir+'\\Fukuchi2017-Running-SequentialAnalysisSummary.csv',
                     index = False)

# %% TODO: sample plots for sequential analysis...

#Sample violin plot of sequential analysis for number of gait cycles
fig, ax = plt.subplots(nrows = len(stabilityLevels), ncols = 1, figsize = (9,8))

#Plot violin plot for first variable
#Y-axis represents number of gait cycles, X-axis represents trial type
#Both 0D and 1D variables are plotted

#For a single variable

# for ss in range(len(stabilityLevels)):

#     sns.violinplot(data = df_seqResults.loc[(df_seqResults['analysisVar'] == analysisVar[0]) &
#                                             (df_seqResults['stabilityThreshold'] == stabilityLevels[ss])],
#                    x = 'trialID', y = 'stabilityGC', hue = 'varType',
#                    split = True, inner = None,
#                    palette = 'colorblind', linewidth = 0,
#                    ax = ax[ss])

#For a single trial type

#### Probably better...

for ss in range(len(stabilityLevels)):

    sns.violinplot(data = df_seqResults.loc[(df_seqResults['trialID'] == trialList[0]) &
                                            (df_seqResults['stabilityThreshold'] == stabilityLevels[ss])],
                   x = 'analysisVar', y = 'stabilityGC', hue = 'varType',
                   split = True, inner = None,
                   palette = 'colorblind', linewidth = 0,
                   ax = ax[ss])
    
    
#### TODO: pretty-up figures...export dataframe to for markdown use...

#Sample box plot of data for knee angle

#####
#####
##### TODO: FIX UP!
##### A dot plot with 95% CI's and outliers might be more appropriate?
#####
#####

#Initialise figure
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,3.5))

#Plot the 0.25 SD bandwidth
##### TODO: add annotation?
ax.axhline(y = 0.25, linewidth = 1, linestyle = '--', color = 'grey')
ax.axhline(y = -0.25, linewidth = 1, linestyle = '--', color = 'grey')

#Extract current trial and analysis variable dataframe
df_currSeq = df_seqAnalysis.loc[(df_seqAnalysis['trialID'] == trialList[tt]) &
                                (df_seqAnalysis['analysisVar'] == analysisVar[vv]),]

#Plot boxplot with Seaborn
sns.boxplot(data = df_currSeq, x = 'nGC', y = 'seqVal',
            whis = [0,100], palette = 'colorblind', hue = 'varType',
            ax = ax)

#### Boxplots demonstrate the number of gait cycles for *EVERYONE* to get under
#### the 0.25 SD threshold for X consecutive cycles --- but this will vary from
#### person to person...
####
#### Also consider the appropriateness of a 0.25 SD threshold with the absolute
#### maximum value here --- it could be quite sensitive and may be valid to calculate
#### the number of cycles fro variable thresholds...
####
#### Still takes a long time for peak values to come under the 0.25 SD threshold
#### Could consider taking absolute of these peak SD sequential values too for
#### consistency with the 1D variables...

# %% BELOW = POTENTIALLY OLD / ALREADY DONE...

# %% RQ2: Varying cycle number comparison to 'ground truth' mean
    
# This analysis compares a sample of an individuals treadmill bout to their 
# entire treadmill bout (i.e. 'ground truth'). Given we don't know exactly what
# an individuals 'true' kinematics are, we take the entire treadmill bout as this
# 'truth' value --- and this is supported by the sequential analysis results
# where it appears most (if not all) have a pretty stable mean. 
#
# The sampled mean (with varying number of gait cycles) is compared to our 'ground
# truth' mean, and given these come from the same treadmill running bout we effectively
# want to see no statistical difference. With the alpha level set at 0.05, we do
# however consider a false-positive rate of 5% to come up.
#
# Here we consider 0D peak variables as week as the 1D curves.


#Set a dictionary to store findings of each iteration in
##### TODO
    #Include some sort of standardised error/effect of difference???
groundTruthDict = {'extractNo': [], 'trialID': [], 'varType': [],
                   'rejectH0': [], 'pVal': [],
                   'analysisVar': [],
                   'groundTruth': [], 'extract': [],
                   'groundTruth_m': [], 'extract_m': [], 'meanAbsError': []}

#Loop through the different trial types
for tt in range(len(trialList)):
    
    #Extract the dataframe for the current trial ID
    df_currTrial = df_samples.loc[df_samples['trialID'] == trialList[tt],]
    
    #Loop through analysis variables
    for vv in range(len(analysisVar)):
        
        #Calculate the 'ground truth' values based on all of the participants
        #gait cycles
        
        #Set the 'ground truth' mean array
        groundTruth_0D = np.zeros((len(subList),1))
        groundTruth_1D = np.zeros((len(subList),101))
        
        #Loop through subjects
        for ii in range(len(subList)):
            
            #Extract the current participants kinematic data relevant to
            #current trial tupe. Get the index corresponding to this in the
            #data dictionary.
            subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
            trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
            currInd = list(set(subInd) & set(trialInd))[0]
            
            #Get the right foot strike indices
            rightFS = dataDict['rightFS'][currInd]
            
            #Extract participants normalised data for the current trial and variable
            normData = normDataDict[trialList[tt]][analysisVar[vv]][ii]
                
            #Calculate the mean of the current subjects normalised data
            #Store in the current ground truths array for SPM1D analysis
            #Also calculate the mean of the peaks here and store in 0D array
            groundTruth_1D[ii,:] = np.mean(normData, axis = 0)
            groundTruth_0D[ii,0] = np.mean(np.max(normData, axis = 1), axis = 0)
            
        #Flatten 0D array
        groundTruth_0D = groundTruth_0D.flatten()
        
        #Loop through the extraction numbers
        for ee in range(len(extractNo)):
            
            #Set current extract number
            currNo = int(extractNo[ee])
            
            #Extract the dataframe for the current extraction number
            df_currExtract = df_currTrial.loc[df_currTrial['extractNo'] == currNo,]

            #Loop through the sampling number
            for ss in range(nSamples):
                
                #Set array to store each subjects datasets for this sample iteration
                #Both 0D and 1D variables
                extract_1D = np.zeros((len(subList),101))
                extract_0D = np.zeros((len(subList),1))
                
                #Loop through subjects and get their data for the current sample
                for ii in range(len(subList)):
                                        
                    #Extract the dataframe that matches the current subject
                    #As part of this also extract the row that matches the current
                    #sampling iteration index
                    df_currSub = df_currExtract.loc[df_currExtract['subID'] == subList[ii],]
                    df_currIter = df_currSub.iloc[ss]
                    
                    #Extract the current participants kinematic data relevant to
                    #current trial tupe. Get the index corresponding to this in the
                    #data dictionary.
                    subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                    trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                    currInd = list(set(subInd) & set(trialInd))[0]
                    
                    #Get the right foot strike indices
                    rightFS = dataDict['rightFS'][currInd]
                    
                    #Extract participants normalised data for the current trial and variable
                    normDataAll = normDataDict[trialList[tt]][analysisVar[vv]][ii]
                    
                    #Set a space to store normalised data for current extraction
                    normData = np.zeros((currNo,101))
                    
                    #Loop through extraction number and get normalised kinematics
                    for nn in range(currNo):
                        
                        #Grab the appropriate gait cycle and append to array
                        normData[nn,:] = normDataAll[df_currIter['footStrikes'][nn]]
                
                    #Calculate the mean of the current subjects normalised data
                    #Store in the current sample iterations array for SPM1D analysis
                    extract_1D[ii,:] = np.mean(normData, axis = 0)
                    extract_0D[ii,0] = np.mean(np.max(normData, axis = 1), axis = 0)
                    
                #Flatten 0D array
                extract_0D = extract_0D.flatten()
                
                #Conduct the SPM1D t-test on this sample
                t = spm1d.stats.ttest_paired(groundTruth_1D, extract_1D)
                ti = t.inference(alpha, two_tailed = True, interp = True)
                
                # #Visuliase
                # #Set-up plot
                # plt.figure(figsize=(8, 3.5))
                # #Plot mean and SD of two samples
                # ax1 = plt.axes((0.1, 0.15, 0.35, 0.8))
                # spm1d.plot.plot_mean_sd(groundTruth_1D, linecolor = 'b', facecolor = 'b')
                # spm1d.plot.plot_mean_sd(extract_1D, linecolor = 'r', facecolor='r')
                # ax1.axhline(y = 0, color = 'k', linestyle=':')
                # ax1.set_xlabel('0-100% Gait Cycle')
                # ax1.set_ylabel(analysisVar[vv])
                # #Plot SPM results
                # ax2 = plt.axes((0.55,0.15,0.35,0.8))
                # ti.plot()
                # ti.plot_threshold_label(fontsize = 8)
                # ti.plot_p_values(size = 10, offsets = [(0,0.3)])
                # ax2.set_xlabel('0-100% Gait Cycle')
                # #Show plot
                # plt.show()
                
                #Calculate mean absolute error of current two curves
                groundTruth_1D_m = np.mean(groundTruth_1D, axis = 0)
                extract_1D_m = np.mean(extract_1D, axis = 0)
                mae_1D = np.mean(abs(groundTruth_1D_m - extract_1D_m))
                
                #Conduct paired t-test on peak values
                t0,p0 = stats.ttest_rel(groundTruth_0D, extract_0D)
                if p0 < 0.05:
                    h0reject_0D = True
                else:
                    h0reject_0D = False
                    
                #Calculate means and absolute error for 0D variable
                groundTruth_0D_m = np.mean(groundTruth_0D, axis = 0)
                extract_0D_m = np.mean(extract_0D, axis = 0)
                mae_0D = np.mean(abs(groundTruth_0D_m - extract_0D_m))
                
                #Collate results from this sampling iteration into dictionary
                #1D
                groundTruthDict['extractNo'].append(currNo)
                groundTruthDict['trialID'].append(trialList[tt])
                groundTruthDict['varType'].append('1D')
                groundTruthDict['rejectH0'].append(ti.h0reject)
                groundTruthDict['pVal'].append(ti.p) #note there are no p-values for non-statistically significant results
                groundTruthDict['analysisVar'].append(analysisVar[vv])
                groundTruthDict['groundTruth'].append(groundTruth_1D)
                groundTruthDict['extract'].append(extract_1D)
                groundTruthDict['groundTruth_m'].append(groundTruth_1D_m)
                groundTruthDict['extract_m'].append(extract_1D_m)
                groundTruthDict['meanAbsError'].append(mae_1D)
                #0D
                groundTruthDict['extractNo'].append(currNo)
                groundTruthDict['trialID'].append(trialList[tt])
                groundTruthDict['varType'].append('0D')
                groundTruthDict['rejectH0'].append(h0reject_0D)
                groundTruthDict['pVal'].append(p0)
                groundTruthDict['analysisVar'].append(analysisVar[vv])
                groundTruthDict['groundTruth'].append(groundTruth_0D)
                groundTruthDict['extract'].append(extract_0D)
                groundTruthDict['groundTruth_m'].append(groundTruth_0D_m)
                groundTruthDict['extract_m'].append(extract_0D_m)
                groundTruthDict['meanAbsError'].append(mae_0D)
                
                #Print confirmation
                print('Completed ground truth comparison '+str(ss+1)+' of '+str(nSamples)+' for '+
                      str(currNo)+' gait cycles of '+analysisVar[vv]+' from '+
                      trialList[tt])

#Stash the ground truth comparisons for speedier use later
savePickleDict(groundTruthDict, storeAnalysisDir+'\\Fukuchi2017-Running-groundTruthDict.pbz2')
# groundTruthDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-groundTruthDict.pbz2')

#Convert dictionary to a dataframe
df_groundTruthComp = pd.DataFrame.from_dict(groundTruthDict)


##### TODO: calculations...

#Visualise null hypothesis rejection rate across gait cycles

#### TODO: loop through trial type

#Initialise figure
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (6,7))

#Plot the expected false positive rate
ax[0].axhline(y = 0.05, linewidth = 1, linestyle = '--', color = 'grey')
ax[1].axhline(y = 0.05, linewidth = 1, linestyle = '--', color = 'grey')

#Loop through variables
for vv in range(len(analysisVar)):
# for vv in range(0,3):

    #Initialise arrays to store H0 reject rate vs. gait cycle number
    X = np.zeros((len(extractNo),1))
    Y_0D = np.zeros((len(extractNo),1))
    Y_1D = np.zeros((len(extractNo),1))
    
    #Loop through extraction number to get the count and H0 rejection rate
    for ee in range(len(extractNo)):
        
        #Set extraction number in array
        X[ee,0] = extractNo[ee]
        
        #Sum the number of times H0 was rejected and add to array
        #Both 0D and 1D variables here
        Y_0D[ee,0] = len(df_groundTruthComp.loc[(df_groundTruthComp['trialID'] == trialList[tt]) &
                                                (df_groundTruthComp['analysisVar'] == analysisVar[vv]) &
                                                (df_groundTruthComp['varType'] == '0D') &
                                                (df_groundTruthComp['extractNo'] == extractNo[ee]) &
                                                (df_groundTruthComp['rejectH0'] == True),['rejectH0']]) / nSamples
        Y_1D[ee,0] = len(df_groundTruthComp.loc[(df_groundTruthComp['trialID'] == trialList[tt]) &
                                                (df_groundTruthComp['analysisVar'] == analysisVar[vv]) &
                                                (df_groundTruthComp['varType'] == '1D') &
                                                (df_groundTruthComp['extractNo'] == extractNo[ee]) &
                                                (df_groundTruthComp['rejectH0'] == True),['rejectH0']]) / nSamples
    
    #Plot data
    ax[0].plot(X, Y_0D, color = analysisCol[vv], marker = analysisSym[vv],
               label = analysisLabels[vv])
    ax[1].plot(X, Y_1D, color = analysisCol[vv], marker = analysisSym[vv],
               label = analysisLabels[vv])

# %% RQ3: Comparing samples from different sections of trial

##### TODO: this is repetitive of the above sample cycling, so perhaps encase
##### both within same loop to speed up process...

#Set a dictionary to store findings of each iteration in
##### TODO
    #Include some sort of standardised error/effect of difference???
    #Should do this for all - SMD for 0D variables; peak SMD for 1D (i.e. biggest difference)???
compDict = {'extractNo': [], 'trialID': [], 'varType': [],
            'rejectH0': [], 'pVal': [],
            'analysisVar': [],
            'Y1': [], 'Y2': [],
            'Y1_m': [], 'Y2_m': [], 'meanAbsError': []}

#Loop through the different trial types
for tt in range(len(trialList)):
    
    #Extract the dataframe for the current trial ID
    df_currTrial = df_samples.loc[df_samples['trialID'] == trialList[tt],]
    
    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through the extraction numbers
        for ee in range(len(extractNoDual)):
            
            #Set current extract number
            currNo = int(extractNoDual[ee])
            
            #Extract the dataframe for the current extraction number
            df_currExtract = df_currTrial.loc[df_currTrial['extractNo'] == currNo,]
            
            #Loop through the sampling number
            for ss in range(nSamples):
                
                #Set array to store each subjects two datasets for this sample iteration
                Y1_1D = np.zeros((len(subList),101))
                Y2_1D = np.zeros((len(subList),101))
                Y1_0D = np.zeros((len(subList),1))
                Y2_0D = np.zeros((len(subList),1))
                
                #Loop through subjects and get their data for the current sample
                for ii in range(len(subList)):
                    
                    #Extract the dataframe that matches the current subject
                    #As part of this also extract the row that matches the current
                    #sampling iteration index
                    df_currSub = df_currExtract.loc[df_currExtract['subID'] == subList[ii],]
                    df_currIter = df_currSub.iloc[ss]
                    
                    #Extract the current participants kinematic data relevant to
                    #current trial tupe. Get the index corresponding to this in the
                    #data dictionary.
                    subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                    trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                    currInd = list(set(subInd) & set(trialInd))[0]
                    
                    #Get the right foot strike indices
                    rightFS = dataDict['rightFS'][currInd]
                    
                    #Extract participants normalised data for the current trial and variable
                    normDataAll = normDataDict[trialList[tt]][analysisVar[vv]][ii]
                    
                    #Set a space to store normalised data for current extraction
                    normData1 = np.zeros((currNo,101))
                    normData2 = np.zeros((currNo,101))
                    
                    #Loop through extraction number and get normalised kinematics
                    for nn in range(currNo):
                        
                        #Grab the appropriate gait cycle and append to arrays
                        normData1[nn,:] = normDataAll[df_currIter['footStrikes1'][nn]]
                        normData2[nn,:] = normDataAll[df_currIter['footStrikes2'][nn]]
                
                    #Calculate the mean of the current subjects normalised data
                    #Store in the current sample iterations array for SPM1D analysis
                    #Also store the peak value for the 0D variable
                    Y1_1D[ii,:] = np.mean(normData1, axis = 0)
                    Y1_0D[ii,0] = np.mean(np.max(normData1, axis = 1), axis = 0)
                    Y2_1D[ii,:] = np.mean(normData2, axis = 0)
                    Y2_0D[ii,0] = np.mean(np.max(normData2, axis = 1), axis = 0)
                
                #Flatten 0D array
                Y1_0D = Y1_0D.flatten()
                Y2_0D = Y2_0D.flatten()
                
                #Conduct the SPM1D t-test on this sample
                t = spm1d.stats.ttest_paired(Y1_1D, Y2_1D)
                ti = t.inference(alpha, two_tailed = True, interp = True)
                
                # #Visuliase
                # #Set-up plot
                # plt.figure(figsize=(8, 3.5))
                # #Plot mean and SD of two samples
                # ax1 = plt.axes((0.1, 0.15, 0.35, 0.8))
                # spm1d.plot.plot_mean_sd(Y1_1D, linecolor = 'b', facecolor = 'b')
                # spm1d.plot.plot_mean_sd(Y1_1D, linecolor = 'r', facecolor='r')
                # ax1.axhline(y = 0, color = 'k', linestyle=':')
                # ax1.set_xlabel('0-100% Gait Cycle')
                # ax1.set_ylabel(analysisVar[vv])
                # #Plot SPM results
                # ax2 = plt.axes((0.55,0.15,0.35,0.8))
                # ti.plot()
                # ti.plot_threshold_label(fontsize = 8)
                # ti.plot_p_values(size = 10, offsets = [(0,0.3)])
                # ax2.set_xlabel('0-100% Gait Cycle')
                # #Show plot
                # plt.show()
                
                #Calculate mean absolute error of current two curves
                Y1_1D_m = np.mean(Y1_1D, axis = 0)
                Y2_1D_m = np.mean(Y2_1D, axis = 0)
                mae_1D = np.mean(abs(Y1_1D_m - Y2_1D_m))
                
                #Conduct paired t-test on peak values
                t0,p0 = stats.ttest_rel(Y1_0D, Y2_0D)
                if p0 < 0.05:
                    h0reject_0D = True
                else:
                    h0reject_0D = False
                    
                #Calculate means and absolute error for 0D variable
                Y1_0D_m = np.mean(Y1_0D, axis = 0)
                Y2_0D_m = np.mean(Y2_0D, axis = 0)
                mae_0D = np.mean(abs(Y1_0D_m - Y2_0D_m))
                
                #Collate results from this sampling iteration into dictionary
                #1D
                compDict['extractNo'].append(currNo)
                compDict['trialID'].append(trialList[tt])
                compDict['varType'].append('1D')
                compDict['rejectH0'].append(ti.h0reject)
                compDict['pVal'].append(ti.p) #note there are no p-values for non-statistically significant results
                compDict['analysisVar'].append(analysisVar[vv])
                compDict['Y1'].append(Y1_1D)
                compDict['Y2'].append(Y2_1D)
                compDict['Y1_m'].append(Y1_1D_m)
                compDict['Y2_m'].append(Y2_1D_m)
                compDict['meanAbsError'].append(mae_1D)
                #0D
                compDict['extractNo'].append(currNo)
                compDict['trialID'].append(trialList[tt])
                compDict['varType'].append('0D')
                compDict['rejectH0'].append(h0reject_0D)
                compDict['pVal'].append(p0)
                compDict['analysisVar'].append(analysisVar[vv])
                compDict['Y1'].append(Y1_0D)
                compDict['Y2'].append(Y2_0D)
                compDict['Y1_m'].append(Y1_0D_m)
                compDict['Y2_m'].append(Y2_0D_m)
                compDict['meanAbsError'].append(mae_0D)
                
                #Print confirmation
                print('Completed '+str(ss+1)+' of '+str(nSamples)+' for '+
                      str(currNo)+' gait cycles of '+analysisVar[vv]+' from '+
                      trialList[tt])

#Stash the cycle comparisons for speedier use later
savePickleDict(compDict, storeAnalysisDir+'\\Fukuchi2017-Running-compDict.pbz2')
# compDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-compDict.pbz2')
                
#Convert to dataframe
df_compSamples = pd.DataFrame.from_dict(compDict)

# %% UP TO HERE...

#Test false positive comparison rate between samples
rejectH0 = df_compSamples.loc[(df_compSamples['extractNo'] == extractNoDual[0]) &
                              (df_compSamples['trialID'] == trialList[0]) &
                              (df_compSamples['varType'] == '0D'),
                              ['rejectH0']]['rejectH0'].to_numpy()

np.sum(rejectH0) / len(rejectH0)



# %% RQ4: Cycle sampling and number effect on refuting null hypothesis

# This section examines the effect of gait cycle number and sampling on the findings
# from null hypothesis testing, specifically the effect of speed on gait biomechanics.
# The original work of Fukuchi et al. (2017) examined this question, and found 
# a series of biomechanical variables are impacted by gait speed. Here we iteratively
# replicate this hypothesis testing with varying numbers and differently sampled
# gait cycles contributing to participant data. The theory being tested here is
# how does the number and selection of gait cycles impact the answers to our hypothesis
# testing.

##### TODO: considerations around 0D vs. 1D variables, ANOVA vs. t-test

##### To start with, replicate the Fukuchi et al. analysis...

##### Start with peak knee flexion as a test
vv = 3




#Set dictionary to store 0D ANOVA results
anovaDict = {'extractNo': [], 'analysisVar': [], 'startPoint': [],
             'analysisData': [], 'mean': [], 'sd': [],
             'aovrmResults': [], 'F': [], 'p': [], 'rejectH0': []}

#Set dictionary to store 0D post-hoc pairwise results
pairwiseDict = {'extractNo': [], 'analysisVar': [], 'startPoint': [],
                'comparison': [], 'val0D': [], 'mean': [], 'sd': [],
                't': [], 'p': [], 'rejectH0': []}

#Loop through the extraction numbers and run analyses
for ee in range(len(extractNo)):
    
    #Set seed for consistent subsequent random sampling
    random.seed(12345)
    
    #Set current extract number
    currNo = int(extractNo[ee])
    
    #Loop through the sampling number
    for ss in range(nSamples):
        
        #Set dictionary to store data into for current sample iteration
        analysisDict = {'subID': [], 'val0D': [], 'speed': []}
        
        #Select the starting point to take gait cycles from based on current
        #extraction number
        
        #Create an appropriate list to select from based on min. gait cycles
        gcStarts = np.linspace(0,minGC,minGC+1)
        #Remove the final X values relative to current extraction number, as
        #these won't leave enough to grab
        goZone = gcStarts[0:int(-extractNo[ee])]
        
        #Choose the starting point
        startPoint = int(random.choice(goZone))
        
        # #Set an array to store each subjects trial datasets in for this iteration
        # ##### NOTE: for 0D variable
        # analysisVals = np.zeros((len(subList),len(trialList)))

        #Loop through subjects and get their data for the current sample iteration
        for ii in range(len(subList)):
            
            #Loop through trials
            for tt in range(len(trialList)):
                                
                #Extract the array for the current participant and trial
                currData = normDataDict[trialList[tt]][ii]
                
                #Extract the current selection of gait cycles based on the starting
                #point and current extraction number
                currExtraction = currData[startPoint:startPoint+currNo,:]
                
                #Calculate the peak value across each cycle and get the mean peak
                #Append this to the analysis dictionary in appropriate spot
                ##### NOTE: for 0D variable
                analysisDict['val0D'].append(np.mean(np.max(currExtraction, axis = 1), axis = 0))
                
                #Append additional factors to dictionary
                analysisDict['subID'].append(ii)
                analysisDict['speed'].append(trialList[tt])
        
        #Convert dictionary to dataframe for ANOVA
        df_analysis = pd.DataFrame.from_dict(analysisDict)
        
        #Run and fit the one-way repeated measures ANOVA
        aovrm = AnovaRM(df_analysis, 'val0D', 'subID', within = ['speed'])
        aovrmResults = aovrm.fit()
        # print(aovrmResults)
        
        #Extract ANOVA results for current iteration
        F = aovrmResults.anova_table['F Value']['speed']
        p = aovrmResults.anova_table['Pr > F']['speed']
        if p < 0.05:
            rejectH0_anova = True
        else:
            rejectH0_anova = False
        
        #Store ANOVA results in dictionary
        anovaDict['extractNo'].append(currNo)
        anovaDict['analysisVar'].append(analysisVar[vv])
        anovaDict['startPoint'].append(startPoint)
        anovaDict['analysisData'].append(df_analysis)
        anovaDict['mean'].append(df_analysis.groupby('speed').mean()['val0D'])
        anovaDict['sd'].append(df_analysis.groupby('speed').std()['val0D'])
        anovaDict['aovrmResults'].append(aovrmResults)
        anovaDict['F'].append(F)
        anovaDict['p'].append(p)
        anovaDict['rejectH0'].append(rejectH0_anova)
        
        #Get and run post-hoc if appropriate            
        #Loop through pairwise trial comparisons
        for pp in range(len(trialList)-1):
            for qq in range(pp+1,len(trialList)):
                
                #Extract arrays to compare
                y1 = df_analysis.loc[df_analysis['speed'] == trialList[pp],'val0D'].to_numpy()
                y2 = df_analysis.loc[df_analysis['speed'] == trialList[qq],'val0D'].to_numpy()
                
                if rejectH0_anova:
                
                    #Compare
                    t0,p0 = stats.ttest_rel(y1, y2)
                    
                #Append results to dictionary
                pairwiseDict['extractNo'].append(currNo)
                pairwiseDict['analysisVar'].append(analysisVar[vv])
                pairwiseDict['startPoint'].append(startPoint)
                pairwiseDict['comparison'].append([trialList[pp],trialList[qq]])
                pairwiseDict['val0D'].append([y1,y2])
                pairwiseDict['mean'].append([np.mean(y1),np.mean(y2)])
                pairwiseDict['sd'].append([np.std(y1),np.std(y2)])
                if rejectH0_anova:
                    pairwiseDict['t'].append(t0)
                    pairwiseDict['p'].append(p0)
                    if p0 < 0.05:
                        pairwiseDict['rejectH0'].append(True)
                    else:
                        pairwiseDict['rejectH0'].append(False)           
                else:
                    pairwiseDict['t'].append(np.nan)
                    pairwiseDict['p'].append(np.nan)
                    pairwiseDict['rejectH0'].append(np.nan)
                    
        #Print confirmation
        print('Completed 0D speed comparison '+str(ss+1)+' of '+str(nSamples)+' for '+
              str(currNo)+' gait cycles of '+analysisVar[vv])


#### Does the null hypothesis rejection rate need to be contrasted with a power
#### analysis of sorts to discover our true potential discovery rate?????



                
