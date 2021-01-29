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
#Ground truth comparison directory
os.chdir('..\\GroundTruthComp')
truthCompDir = os.getcwd()
#Samples comparison directory
os.chdir('..\\SamplesComp')
samplesCompDir = os.getcwd()
#Difference comparison directory
os.chdir('..\\DifferenceComp')
diffCompDir = os.getcwd()

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
    
# %% Extract data

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
nSamples = 1000
# nSamples = 100      ### REDUCE DURING CODE TESTING

#Setting for whether to sample gait cycles or load the pickle file
#CHANGE THIS TO TRUE IF THE PICKLE (.PBZ2) FILES DON'T EXIST
sampleGaitCycles = False

#Check for normalising data
if sampleGaitCycles:
    
    #Loop through extraction numbers
    for ee in range(len(extractNo)):
        
        #Set dictionary to store gait cycle points in
        sampleDict = {'subID': [], 'trialID': [], 'extractNo': [],
                      'footStrikes': [], 'footStrikes1': [], 'footStrikes2': []}
    
        #Loop through subjects
        for ii in tqdm.tqdm(range(len(subList)),
                            desc = 'Sampling '+str(int(extractNo[ee]))+' gait cycles from '+
                            str(len(subList))+' participants'):
            
            #Loop through trial list
            for tt in range(len(trialList)):
                
                #Get the index for the current subject/trial combo
                subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                currInd = list(set(subInd) & set(trialInd))[0]
            
                #Get the number of gait cycles for this participant
                #Note that this is for the right limb this is what we'll analyse
                nGC = dataDict['rGC_n'][currInd]
                    
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
                # for ss in tqdm.tqdm(range(nSamples),
                #                 desc = 'Sampling '+str(int(extractNo[ee]))+
                #                 ' gait cycles for '+subList[ii]+' from '
                #                 +trialList[tt]):
                for ss in range(nSamples):
                    
                    #Append subject and trial ID names
                    sampleDict['subID'].append(subList[ii])
                    sampleDict['trialID'].append(trialList[tt])
                    
                    #Append extract number details
                    sampleDict['extractNo'].append(int(extractNo[ee]))
                    
                    #Select a random number from the single 'go zone'
                    singlePick = random.choice(goZoneSingle)
                    
                    #Set strikes for current sample in directory
                    footStrikes = list(map(round,list(np.linspace(singlePick,singlePick+int(extractNo[ee]),int(extractNo[ee]+1)))))
                    sampleDict['footStrikes'].append(footStrikes)
                    
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
                        footStrikes1 = list(map(round,list(np.linspace(dualPick1,dualPick1+int(extractNo[ee]),int(extractNo[ee]+1)))))
                        footStrikes2 = list(map(round,list(np.linspace(dualPick2,dualPick2+int(extractNo[ee]),int(extractNo[ee]+1)))))
                        sampleDict['footStrikes1'].append(footStrikes1)
                        sampleDict['footStrikes2'].append(footStrikes2)
                    
                    #Otherwise just set nan's
                    else:
                        
                        sampleDict['footStrikes1'].append(np.nan)
                        sampleDict['footStrikes2'].append(np.nan)
                        
        #Stash the compiled sampling vals for speedier use later
        savePickleDict(sampleDict, storeAnalysisDir+'\\Fukuchi2017-Running-gcSamples-n'+
                       str(nSamples)+'-gc'+str(int(extractNo[ee]))+'.pbz2')
            
        #Print confirmation for subject
        print('\nSampling completed for '+str(ee+1)+' of '+str(len(extractNo))+' gait cycle numbers.')
        
    #Print confirmation
    print('All gait cycle sampling completed!')
    
else:
    
    #Print out message
    print('Sampling seems to have already been done...')
    
# %% Research questions

# The below section undertakes a series of analyses to answer the specific research
# questions (RQs) for this study.Below cell sections include specific notes on
# the various RQs prior to them being encapsulated within a set series of loops.
# The below analysis and sampling procedures are time intensive, hence the pre
# analysed data is available and can be loaded in by keeping the below variable
# set to False. If analysis needs to be undertaken, set this to True.

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

# %% Work through samples

#Setting for whether to analyse samples
#This is time consuming, so would aoid repeating
analyseSamples = False

#Check to run analysis
if analyseSamples:

    #Calculate total iterations for progressive timing and time left estimates
    totalIter = len(trialList) * len(analysisVar) * len(extractNo)
    
    #Set a current interations variable
    currIter = 0
    
    #Start timer
    t0 = time.time()
    
    #Loop through trial types
    for tt in range(len(trialList)):  
        
        # #Extract the dataframe for the current trial ID
        # df_currTrial = df_samples.loc[df_samples['trialID'] == trialList[tt],]
        
        #Loop through variables
        for vv in range(len(analysisVar)):
                
            #Calculate the 'ground truth' values based on all of the participants
            #gait cycles for the current variables        
            #Set the 'ground truth' mean array
            groundTruth_0D = np.zeros((len(subList)))
            groundTruth_1D = np.zeros((len(subList),101))
            
            #Set the sequential analysis dictionary to fill for current variable (RQ1)
            seqDict = {'nGC': [], 'subID': [], 'trialID': [], 'analysisVar': [],
                       'seqVal': [], 'varType': []}
                    
            #Loop through participants
            for ii in range(len(subList)):
                
                #Get the number of total right foot gait cycles for current participant
                #Get the index corresponding to the current participant and trial type in data dictionary
                subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                currInd = list(set(subInd) & set(trialInd))[0]
                
                #Get the number of right foot strike indices
                nRFS = len(dataDict['rightFS'][currInd])
                
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
                groundTruth_0D[ii] = np.mean(np.max(normDataAll, axis = 1), axis = 0)
                
                #Loop through n+1 number of gait cycles sequentially and assess
                #points relative to +/- 0.25 SD bounds
                for nn in range(1,nRFS-1):
                    
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
                    
            #Save sequential analysis dictionary for current participant
            savePickleDict(seqDict, seqAnalysisDir+'\\Fukuchi2017-Running-seqAnalysis-'+
                            analysisVar[vv]+'-'+trialList[tt]+'.pbz2')
                    
            #Loop through the extraction numbers
            for ee in range(len(extractNo)):
                
                #Set current extract number
                currNo = int(extractNo[ee])
                
                #Set a dictionary to store the ground truth comparisons for
                #current gait cycle number (RQ2)
                groundTruthDict = {'extractNo': [], 'trialID': [], 'varType': [],
                                   'rejectH0': [], 'pVal': [],
                                   'analysisVar': [],
                                   # 'groundTruth': [], 'extract': [],
                                   'groundTruth_m': [], 'extract_m': [],
                                   'meanAbsError': [], 'peakAbsError': [], 'effectSize': []}
                
                #Set a dictionary to store findings of comparing different samples
                #of gait cycles for th current gait cycle number (RQ3)
                if currNo <= max(extractNoDual):
                    compDict = {'extractNo': [], 'trialID': [], 'varType': [],
                                'rejectH0': [], 'pVal': [],
                                'analysisVar': [],
                                # 'Y1': [], 'Y2': [],
                                'Y1_m': [], 'Y2_m': [],
                                'meanAbsError': [], 'peakAbsError': [], 'effectSize': []}
                
                #Set dictionary to store extracted values in for 0D ANOVA (RQ4)
                anovaDataDict = {'sampleNo': [], 'extractNo': [],
                                 'vals': [], 'varType': [], 
                                 'speed': []}
                
                # #Extract the dataframe for the current extraction number
                # df_currExtract = df_currTrial.loc[df_currTrial['extractNo'] == currNo,]
                
                # start = time.perf_counter()
                
                #Load the sampling dictionary for the current gait cycle number
                sampleDict = loadPickleDict(storeAnalysisDir+'\\Fukuchi2017-Running-gcSamples-n'+
                                            str(nSamples)+'-gc'+str(currNo)+'.pbz2')
                
                #Convert dictionary to a dataframe
                df_sample = pd.DataFrame.from_dict(sampleDict)
                
                # finish = time.perf_counter()
                # print(f'Load data done in {round(finish-start,5)} seconds(s)')
                
                #Loop through the sampling number
                for ss in tqdm.tqdm(range(nSamples),
                                          desc = 'Analysing '+str(currNo)+' gait cycles for '+
                                          analysisVar[vv]+' from '+trialList[tt]):
                    
                    #Set array to store the extracted datasets for this sample iteration
                    #that will be compared to the ground truth dataset
                    #Both 0D and 1D variables
                    extract_1D = np.zeros((len(subList),101))
                    extract_0D = np.zeros((len(subList)))
                    
                    #Set array to store the extracted datasets for this sample iteration
                    #that will be compared to one another
                    #Both 0D and 1D variables
                    #Not that this analysis is only undertaken if the current gait
                    #cycle extraction number is relevant for comparing the two
                    if currNo <= max(extractNoDual):
                        Y1_1D = np.zeros((len(subList),101))
                        Y2_1D = np.zeros((len(subList),101))
                        Y1_0D = np.zeros((len(subList)))
                        Y2_0D = np.zeros((len(subList)))
                    
                    # start = time.perf_counter()
                    
                    #Get the index corresponding to the current trial type for all participants
                    currInd = df_sample.loc[(df_sample['trialID'] == trialList[tt]),].index[
                        list(np.linspace(0, nSamples * (len(subList) - 1), len(subList), dtype = int) + ss)
                        ]
                    
                    #Extract normalised data for the current trial and variable for all participants
                    normDataAll = normDataDict[trialList[tt]][analysisVar[vv]]
                    
                    #Get set of foot strikes for all participants and current sample
                    footStrikes = df_sample['footStrikes'][currInd]
                    if currNo <= maxExtractDual:
                        footStrikes1 = df_sample['footStrikes1'][currInd]
                        footStrikes2 = df_sample['footStrikes2'][currInd]
                        
                    #Get the series of normalised kinematics
                    #Also includes mean/max calculations and appending to array
                    for ii in range(len(subList)):
                        extract_1D[ii,:] = np.mean(normDataAll[ii][footStrikes[currInd[ii]][0]:footStrikes[currInd[ii]][-1]], axis = 0)
                        extract_0D[ii] = np.mean(np.max(normDataAll[ii][footStrikes[currInd[ii]][0]:footStrikes[currInd[ii]][-1]], axis = 1), axis = 0)
                        if currNo <= maxExtractDual:
                            Y1_1D[ii,:] = np.mean(normDataAll[ii][footStrikes1[currInd[ii]][0]:footStrikes1[currInd[ii]][-1]], axis = 0)
                            Y1_0D[ii] = np.mean(np.max(normDataAll[ii][footStrikes1[currInd[ii]][0]:footStrikes1[currInd[ii]][-1]], axis = 1), axis = 0)
                            Y2_1D[ii,:] = np.mean(normDataAll[ii][footStrikes2[currInd[ii]][0]:footStrikes2[currInd[ii]][-1]], axis = 0)
                            Y2_0D[ii] = np.mean(np.max(normDataAll[ii][footStrikes2[currInd[ii]][0]:footStrikes2[currInd[ii]][-1]], axis = 1), axis = 0)
     
                    # finish = time.perf_counter()
                    # print(f'Subject loop done in {round(finish-start,5)} seconds(s)')
                    
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
                    
                    #Calculate errors and effects between current two curves
                    groundTruth_1D_m = np.mean(groundTruth_1D, axis = 0)
                    extract_1D_m = np.mean(extract_1D, axis = 0)
                    mae_1D_RQ2 = np.mean(abs(groundTruth_1D_m - extract_1D_m))
                    pae_1D_RQ2 = np.max(abs(groundTruth_1D_m - extract_1D_m))
                    cohens_1D_RQ2 = calcCohensD_1D(groundTruth_1D, extract_1D)
                    
                    #Conduct paired t-test on peak values
                    t0_RQ2,p0_RQ2 = stats.ttest_rel(groundTruth_0D, extract_0D)
                    if p0_RQ2 < 0.05:
                        h0reject_0D_RQ2 = True
                    else:
                        h0reject_0D_RQ2 = False
                        
                    #Calculate errors and effect between current 0D variables
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
                        
                        #Calculate errors and effects between two sampled curves
                        Y1_1D_m = np.mean(Y1_1D, axis = 0)
                        Y2_1D_m = np.mean(Y2_1D, axis = 0)
                        mae_1D_RQ3 = np.mean(abs(Y1_1D_m - Y2_1D_m))
                        pae_1D_RQ3 = np.max(abs(Y1_1D_m - Y2_1D_m))
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
                    # groundTruthDict['groundTruth'].append(groundTruth_1D)
                    # groundTruthDict['extract'].append(extract_1D)
                    groundTruthDict['groundTruth_m'].append(groundTruth_1D_m)
                    groundTruthDict['extract_m'].append(extract_1D_m)
                    groundTruthDict['meanAbsError'].append(mae_1D_RQ2)
                    groundTruthDict['peakAbsError'].append(pae_1D_RQ2)
                    groundTruthDict['effectSize'].append(cohens_1D_RQ2)
                    #0D
                    groundTruthDict['extractNo'].append(currNo)
                    groundTruthDict['trialID'].append(trialList[tt])
                    groundTruthDict['varType'].append('0D')
                    groundTruthDict['rejectH0'].append(h0reject_0D_RQ2)
                    groundTruthDict['pVal'].append(p0_RQ2)                
                    groundTruthDict['analysisVar'].append(analysisVar[vv])
                    # groundTruthDict['groundTruth'].append(groundTruth_0D)
                    # groundTruthDict['extract'].append(extract_0D)
                    groundTruthDict['groundTruth_m'].append(groundTruth_0D_m)
                    groundTruthDict['extract_m'].append(extract_0D_m)
                    groundTruthDict['meanAbsError'].append(mae_0D_RQ2)
                    groundTruthDict['peakAbsError'].append(np.nan) #variable isn't applicable
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
                        # compDict['Y1'].append(Y1_1D)
                        # compDict['Y2'].append(Y2_1D)
                        compDict['Y1_m'].append(Y1_1D_m)
                        compDict['Y2_m'].append(Y2_1D_m)
                        compDict['meanAbsError'].append(mae_1D_RQ3)
                        compDict['peakAbsError'].append(pae_1D_RQ3)
                        compDict['effectSize'].append(cohens_1D_RQ3)
                        #0D
                        compDict['extractNo'].append(currNo)
                        compDict['trialID'].append(trialList[tt])
                        compDict['varType'].append('0D')
                        compDict['rejectH0'].append(h0reject_0D_RQ3)
                        compDict['pVal'].append(p0_RQ3)                
                        compDict['analysisVar'].append(analysisVar[vv])
                        # compDict['Y1'].append(Y1_0D)
                        # compDict['Y2'].append(Y2_0D)
                        compDict['Y1_m'].append(Y1_0D_m)
                        compDict['Y2_m'].append(Y2_0D_m)
                        compDict['meanAbsError'].append(mae_0D_RQ3)
                        compDict['peakAbsError'].append(np.nan) #variable isn't applicable
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
                    
                #Stash dictionaries for current gait cycle number
                #Ground truth comparison
                savePickleDict(groundTruthDict, truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                                str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                trialList[tt]+'.pbz2')
                #Samples comparison
                if currNo <= max(extractNoDual):
                    savePickleDict(compDict, samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                                    str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                    trialList[tt]+'.pbz2')
                #ANOVA dictionary
                savePickleDict(anovaDataDict, diffCompDir+'\\Fukuchi2017-Running-anovaData-n'+
                                str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                trialList[tt]+'.pbz2')
                
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
                    estHours = int(estRemTime)
                    estMins = int(np.round((estRemTime*60) % 60))
                    print('Estimated analysis time remaining: %d hours & %d minutes\n' % (estHours, estMins))

else:
    
    #Print message
    print('Samples already analysed...')

# %% Summarise sequential analysis data

# Here we work through the sequential analysis data extracted from the previous
# sampling procedure to determine the point of 'stability' across the various
# trial types and variables

#Determine the point of 'stability' across participants and variables at different
#standard deviation thresholds. The works mentioned earlier use a 0.25 SD threshold,
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
        
        #Load in the dictionary associated with current trial and variable
        seqDict = loadPickleDict(seqAnalysisDir+'\\Fukuchi2017-Running-seqAnalysis-'+
                                 analysisVar[vv]+'-'+trialList[tt]+'.pbz2')
        
        #Convert to dataframe
        df_seqAnalysis = pd.DataFrame.from_dict(seqDict)
        
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
savePickleDict(seqResultsDict, seqAnalysisDir+'\\Fukuchi2017-Running-seqResults.pbz2')

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
savePickleDict(seqSummaryDict, storeAnalysisDir+'\\Fukuchi2017-Running-seqSummary.pbz2')

#Convert to dataframe
df_seqSummary = pd.DataFrame.from_dict(seqSummaryDict)

#Export sequential analysis results to file
df_seqSummary.to_csv(seqAnalysisDir+'\\Fukuchi2017-Running-SequentialAnalysisSummary.csv',
                     index = False)

##### TODO: sample plots for sequantial analysis data...

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

# %% Collate comparisons to 'ground truth' mean (RQ2 cont.)

##### TODO: consider how to present statistical comparison results?

#Set a list for collating max error values for reporting purposes
##### TODO: add 0D comparisons...
maxMAE_1D = []
maxPAE_1D = []
maxEffect_1D = []

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through extraction numbers
        for ee in range(len(extractNo)):
            
            #Set current extraction number
            currNo = int(extractNo[ee])
        
            #Load data dictionary for current extraction number
            groundTruthDict = loadPickleDict(truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                                             str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                             trialList[tt]+'.pbz2')
            
            #Convert to dataframe.
            #If first iteration, set as the overall dataframe
            if ee == 0: 
                df_groundTruthComp =  pd.DataFrame.from_dict(groundTruthDict)
            else:
                #Convert to generic dataframe variable and bind to remaining
                df =  pd.DataFrame.from_dict(groundTruthDict)
                df_groundTruthComp = pd.concat([df_groundTruthComp,df])
        
        #Get summary statistics for error values and effect sizes
        
        #Mean absolute error
        groupedMAE = df_groundTruthComp.groupby(['extractNo','varType'])['meanAbsError'].agg(['mean', 'count', 'std'])
        #Effect size
        effectMAE =  df_groundTruthComp.groupby(['extractNo','varType'])['effectSize'].agg(['mean', 'count', 'std'])
        #Peak absolute error
        groupedPAE = df_groundTruthComp.loc[df_groundTruthComp['varType'] == '1D',].groupby(['extractNo','varType'])['peakAbsError'].agg(['mean', 'count', 'std'])
                
        #Loop through data and calculate confidence intervals for errors
        
        #Mean absolute error
        ci95 = []
        #Calculate confidence intervals
        for kk in groupedMAE.index:
            m, c, s = groupedMAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        groupedMAE['ci95'] = ci95
        
        #Effect size
        ci95 = []
        #Calculate confidence intervals
        for kk in effectMAE.index:
            m, c, s = effectMAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        effectMAE['ci95'] = ci95
        
        #Peak absolute error
        
        #Calculate confidence intervals
        ci95 = []
        for kk in groupedPAE.index:
            m, c, s = groupedPAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        groupedPAE['ci95'] = ci95
        
        #Export summary data to file
        groupedMAE.to_csv(truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                          str(nSamples)+'-'+analysisVar[vv]+'-'+
                          trialList[tt]+'_meanAbsErrorSummary.csv')
        groupedPAE.to_csv(truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                          str(nSamples)+'-'+analysisVar[vv]+'-'+
                          trialList[tt]+'_peakAbsErrorSummary.csv')
        effectMAE.to_csv(truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                         str(nSamples)+'-'+analysisVar[vv]+'-'+
                         trialList[tt]+'_effectSizeErrorSummary.csv')
        
        #Print out summary averages
        #0D
        
        #1D
        maxMAE_1D.append([np.max(groupedMAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        print(f'Maximum 1D mean absolute error for {analysisVar[vv]} in {trialList[tt]}: {np.round(maxMAE_1D[-1][0],2)}')
        maxPAE_1D.append([np.max(groupedPAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        print(f'Maximum 1D peak absolute error for {analysisVar[vv]} in {trialList[tt]}: {np.round(maxPAE_1D[-1][0],2)}')
        maxEffect_1D.append([np.max(np.abs(effectMAE['mean'].to_numpy())),analysisVar[vv],trialList[tt]])
        print(f'Maximum 1D mean absolute error effect size for {analysisVar[vv]} in {trialList[tt]}: {np.round(maxEffect_1D[-1][0],2)}')



#### Create a figure for ISB abstract that does all variables combined for mean
#### and peak error (maybe just peak) - gait cycles X axis, error on Y axis when
#### compared to the 'global mean' 

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through extraction numbers
        for ee in range(len(extractNo)):
            
            #Set current extraction number
            currNo = int(extractNo[ee])
        
            #Load data dictionary for current extraction number
            groundTruthDict = loadPickleDict(truthCompDir+'\\Fukuchi2017-Running-groundTruthComp-n'+
                                             str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                             trialList[tt]+'.pbz2')
            
            #Convert to dataframe.
            #If first iteration, set as the overall dataframe
            if ee == 0 and tt == 0 and vv == 0: 
                df_groundTruthComp =  pd.DataFrame.from_dict(groundTruthDict)
            else:
                #Convert to generic dataframe variable and bind to remaining
                df =  pd.DataFrame.from_dict(groundTruthDict)
                df_groundTruthComp = pd.concat([df_groundTruthComp,df])


#Set plot parameters
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['legend.fontsize'] = 12
rcParams['legend.title_fontsize'] = 12
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

#Set figure
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3.5*3,2*3))

#Create plot
h = sns.pointplot(x = 'extractNo', y = 'peakAbsError', hue = 'trialID',
                  data = df_groundTruthComp.loc[df_groundTruthComp['varType'] == '1D',],
                  dodge = True, markers = 's', scale = 0.7, errwidth = 1.5,
                  palette = 'colorblind', ax = ax)
              
#Axes settings
#Y-limits
ax.set_ylim([0,1.2])
#X-tick labels
ax.set_xticks([0,5,10,15,20,25])
ax.set_xlim([0-1,25+1])
#Y-label
ax.set_ylabel('Peak Absolute Error (\u00b0)')
#X-label
ax.set_xlabel('No. of Gait Cycles')

#Legend Settings
#Labels
h, l = ax.get_legend_handles_labels()
l = ['2.5 m\u00b7s$^{-1}$', '3.5 m\u00b7s$^{-1}$', '4.5 m\u00b7s$^{-1}$']
plt.legend(h,l)
#Title
ax.get_legend().set_title('Running Speed')

#Layout
plt.tight_layout()

#Save this to ISB abstract directory for now
plt.savefig('C:\\Users\\aafox\\OneDrive - Deakin University\\RESEARCH\\Conferences\\2021\\ISB 2021\\Abstract\\groundTruthComp_peakAbsErr.png',
            dpi = 300, format = 'png')

#Close plot
plt.close()



##### TEST PLOTS...

#Sample boxplot?
sns.boxplot(x = 'extractNo', y = 'meanAbsError', data = df_groundTruthComp,
            hue = 'varType')

#Sample pointplot?
sns.pointplot(x = 'extractNo', y = 'meanAbsError', data = df_groundTruthComp,
              hue = 'varType', dodge = True, join = False)







#Visualise null hypothesis rejection rate across gait cycles (RQ2)

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


# %% ...

# %% Collate sample comparisons (RQ3 cont.)

##### TODO: consider how to present statistical comparison results?

#Set a list for collating max error values for reporting purposes
##### TODO: add 0D comparisons...
maxMAE_1D = []
maxPAE_1D = []
maxEffect_1D = []
minMAE_1D = []
minPAE_1D = []
minEffect_1D = []

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through extraction numbers
        for ee in range(len(extractNoDual)):
            
            #Set current extraction number
            currNo = int(extractNo[ee])
        
            #Load data dictionary for current extraction number
            compDict = loadPickleDict(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                                      str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                      trialList[tt]+'.pbz2')
            
            #Convert to dataframe.
            #If first iteration, set as the overall dataframe
            if ee == 0: 
                df_samplesComp = pd.DataFrame.from_dict(compDict)
            else:
                #Convert to generic dataframe variable and bind to remaining
                df =  pd.DataFrame.from_dict(compDict)
                df_samplesComp = pd.concat([df_samplesComp,df])
        
        #Get summary statistics for error values and effect sizes
        
        #Mean absolute error
        groupedMAE = df_samplesComp.groupby(['extractNo','varType'])['meanAbsError'].agg(['mean', 'count', 'std'])
        #Effect size
        effectMAE =  df_samplesComp.groupby(['extractNo','varType'])['effectSize'].agg(['mean', 'count', 'std'])
        #Peak absolute error
        groupedPAE = df_samplesComp.loc[df_samplesComp['varType'] == '1D',].groupby(['extractNo','varType'])['peakAbsError'].agg(['mean', 'count', 'std'])
                
        #Loop through data and calculate confidence intervals for errors
        
        #Mean absolute error
        ci95 = []
        #Calculate confidence intervals
        for kk in groupedMAE.index:
            m, c, s = groupedMAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        groupedMAE['ci95'] = ci95
        
        #Effect size
        ci95 = []
        #Calculate confidence intervals
        for kk in effectMAE.index:
            m, c, s = effectMAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        effectMAE['ci95'] = ci95
        
        #Peak absolute error
        
        #Calculate confidence intervals
        ci95 = []
        for kk in groupedPAE.index:
            m, c, s = groupedPAE.loc[kk]
            ci95.append([m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)])
        #Append to data object
        groupedPAE['ci95'] = ci95
        
        #Export summary data to file
        groupedMAE.to_csv(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                          str(nSamples)+'-'+analysisVar[vv]+'-'+
                          trialList[tt]+'_meanAbsErrorSummary.csv')
        groupedPAE.to_csv(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                          str(nSamples)+'-'+analysisVar[vv]+'-'+
                          trialList[tt]+'_peakAbsErrorSummary.csv')
        effectMAE.to_csv(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                         str(nSamples)+'-'+analysisVar[vv]+'-'+
                         trialList[tt]+'_effectSizeErrorSummary.csv')
        
        #Print out summary averages
        #0D
        
        #1D
        maxMAE_1D.append([np.max(groupedMAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        minMAE_1D.append([np.min(groupedMAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        print(f'Error range for 1D mean absolute error for {analysisVar[vv]} in {trialList[tt]}: {np.round(minMAE_1D[-1][0],2)} - {np.round(maxMAE_1D[-1][0],2)}')
        maxPAE_1D.append([np.max(groupedPAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        minPAE_1D.append([np.min(groupedPAE['mean'].to_numpy()),analysisVar[vv],trialList[tt]])
        print(f'Error range for 1D peak absolute error for {analysisVar[vv]} in {trialList[tt]}: {np.round(minPAE_1D[-1][0],2)} - {np.round(maxPAE_1D[-1][0],2)}')
        maxEffect_1D.append([np.max(np.abs(effectMAE['mean'].to_numpy())),analysisVar[vv],trialList[tt]])
        minEffect_1D.append([np.min(np.abs(effectMAE['mean'].to_numpy())),analysisVar[vv],trialList[tt]])
        print(f'Error range for 1D mean absolute error effect size for {analysisVar[vv]} in {trialList[tt]}: {np.round(minEffect_1D[-1][0],2)} - {np.round(maxEffect_1D[-1][0],2)}')




#### Create a figure for ISB abstract that does all variables combined for mean
#### and peak error (maybe just peak) - gait cycles X axis, error on Y axis when
#### comparing the two samples

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through extraction numbers
        for ee in range(len(extractNoDual)):
            
            #Set current extraction number
            currNo = int(extractNoDual[ee])
        
            #Load data dictionary for current extraction number
            compDict = loadPickleDict(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                                      str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                      trialList[tt]+'.pbz2')
            
            #Convert to dataframe.
            #If first iteration, set as the overall dataframe
            if ee == 0 and tt == 0 and vv == 0: 
                df_samplesComp = pd.DataFrame.from_dict(compDict)
            else:
                #Convert to generic dataframe variable and bind to remaining
                df =  pd.DataFrame.from_dict(compDict)
                df_samplesComp = pd.concat([df_samplesComp,df])



#Set figure
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3.5*3,2*3))

#Create plot
h = sns.pointplot(x = 'extractNo', y = 'peakAbsError', hue = 'trialID',
                  data = df_samplesComp.loc[df_samplesComp['varType'] == '1D',],
                  dodge = True, markers = 's', scale = 0.7, errwidth = 1.5,
                  palette = 'colorblind', ax = ax)
              
#Axes settings
#Y-limits
ax.set_ylim([0,1.7])
#X-tick labels
ax.set_xticks([0,5,10])
ax.set_xlim([0-1,10+1])
#Y-label
ax.set_ylabel('Peak Absolute Difference (\u00b0)')
#X-label
ax.set_xlabel('No. of Gait Cycles')

#Legend Settings
#Labels
h, l = ax.get_legend_handles_labels()
l = ['2.5 m\u00b7s$^{-1}$', '3.5 m\u00b7s$^{-1}$', '4.5 m\u00b7s$^{-1}$']
plt.legend(h,l)
#Title
ax.get_legend().set_title('Running Speed')

#Layout
plt.tight_layout()

#Save this to ISB abstract directory for now
plt.savefig('C:\\Users\\aafox\\OneDrive - Deakin University\\RESEARCH\\Conferences\\2021\\ISB 2021\\Abstract\\samplesComp_peakAbsErr.png',
            dpi = 300, format = 'png')

#Close figure
plt.close()



# %% Run ANOVA & post-hoc on sampled data (RQ4 cont.)

##### TODO: add summary notes

#Loop through variables
for vv in range(len(analysisVar)):
    
    #Loop through gait cycle extraction numbers
    for ee in range(len(extractNo)):
        
        #Set current extraction number
        currNo = int(extractNo[ee])

        #Set dictionary to store 0D ANOVA results
        anovaDict_0D = {'extractNo': [], 'analysisVar': [],
                        'analysisData': [], 'mean': [], 'sd': [],
                        'aovrmResults': [], 'F': [], 'p': [], 'rejectH0': [],
                        'pairwiseComp': []}

        #Set generic anova dictionary to store each trial in
        aoData = []
        
        #Extract data from different trials for current variable
        for tt in range(len(trialList)):
            
            #Load the ANOVA dictionary for current trial type
            aoData.append(pd.DataFrame.from_dict(loadPickleDict(diffCompDir+'\\Fukuchi2017-Running-anovaData-n'+
                                                                str(nSamples)+'-gc'+str(int(extractNo[ee]))+'-'+
                                                                analysisVar[vv]+'-'+trialList[tt]+'.pbz2')))
    
        #Loop through samples
        for ss in range(nSamples):

            #Set dictionaries to store 0D and 1D anova data
            analysisDict_0D = {'subID': [], 'val': [], 'speed': []}
            # analysisDict_1D = {'subID': [], 'val0D': [], 'speed': []}
            
            #Set dictionary to store 0D post-hoc pairwise results
            pairwiseDict_0D = {'extractNo': [], 'analysisVar': [],
                               'comparison': [], 'val0D': [], 'mean': [], 'sd': [],
                               't': [], 'p': [], 'rejectH0': []}
            
            #Loop through trials and append data into analysis dictionary
            for tt in range(len(trialList)):
            
                #Extract current sample for 0D data
                val0D = aoData[tt].loc[(aoData[tt]['sampleNo'] == ss) &
                                       (aoData[tt]['varType'] == '0D'),['vals']].values[0][0]
                
                #Append values with appropriate identifiers to analysis dictionary
                for ii in range(len(subList)):
                    analysisDict_0D['subID'].append(subList[ii])
                    analysisDict_0D['val'].append(val0D[ii])
                    analysisDict_0D['speed'].append(trialList[tt])
                
            #Convert analysis dictionary to dataframe
            df_analysis_0D = pd.DataFrame.from_dict(analysisDict_0D)
                    
            #Run and fit the one-way repeated measures ANOVA
            aovrm = AnovaRM(df_analysis_0D, 'val', 'subID', within = ['speed'])
            aovrmResults = aovrm.fit()
            # print(aovrmResults)
            
            #Extract null hypothesis logical for current iteration
            if aovrmResults.anova_table['Pr > F']['speed'] < 0.05:
                rejectH0_anova_0D = True
            else:
                rejectH0_anova_0D = False
            
            #Store ANOVA results in dictionary
            anovaDict_0D['extractNo'].append(currNo)
            anovaDict_0D['analysisVar'].append(analysisVar[vv])
            anovaDict_0D['analysisData'].append(df_analysis_0D)
            anovaDict_0D['mean'].append(df_analysis_0D.groupby('speed').mean()['val'])
            anovaDict_0D['sd'].append(df_analysis_0D.groupby('speed').std()['val'])
            anovaDict_0D['aovrmResults'].append(aovrmResults)
            anovaDict_0D['F'].append(aovrmResults.anova_table['F Value']['speed'])
            anovaDict_0D['p'].append(aovrmResults.anova_table['Pr > F']['speed'])
            anovaDict_0D['rejectH0'].append(rejectH0_anova_0D)
            
            #Get and run post-hoc if appropriate            
            #Loop through pairwise trial comparisons
            for pp in range(len(trialList)-1):
                for qq in range(pp+1,len(trialList)):
                    
                    #Extract arrays to compare
                    y1 = df_analysis_0D.loc[df_analysis_0D['speed'] == trialList[pp],'val'].to_numpy()
                    y2 = df_analysis_0D.loc[df_analysis_0D['speed'] == trialList[qq],'val'].to_numpy()
                    
                    if rejectH0_anova_0D:
                    
                        #Compare
                        t0,p0 = stats.ttest_rel(y1, y2)
                        
                    #Append results to dictionary
                    pairwiseDict_0D['extractNo'].append(currNo)
                    pairwiseDict_0D['analysisVar'].append(analysisVar[vv])
                    pairwiseDict_0D['comparison'].append([trialList[pp],trialList[qq]])
                    pairwiseDict_0D['val0D'].append([y1,y2])
                    pairwiseDict_0D['mean'].append([np.mean(y1),np.mean(y2)])
                    pairwiseDict_0D['sd'].append([np.std(y1),np.std(y2)])
                    if rejectH0_anova_0D:
                        pairwiseDict_0D['t'].append(t0)
                        pairwiseDict_0D['p'].append(p0)
                        if p0 < 0.05:
                            pairwiseDict_0D['rejectH0'].append(True)
                        else:
                            pairwiseDict_0D['rejectH0'].append(False)           
                    else:
                        pairwiseDict_0D['t'].append(np.nan)
                        pairwiseDict_0D['p'].append(np.nan)
                        pairwiseDict_0D['rejectH0'].append(np.nan)
                        
            #Append pairwise comparisons to ANOVA dictionary
            anovaDict_0D['pairwiseComp'].append(pairwiseDict_0D)
                        
            #Print confirmation
            print('Completed 0D speed comparison '+str(ss+1)+' of '+str(nSamples)+' for '+
                  str(currNo)+' gait cycles of '+analysisVar[vv])


##### TODO: save anova data dictionaries...

#### Does the null hypothesis rejection rate need to be contrasted with a power
#### analysis of sorts to discover our true potential discovery rate?????

# %% ...

##### TODO: focus analysis on magnitude of error and effect size, over just pure H0 rejection rate

# %% # %% BELOW = POTENTIALLY OLD / ALREADY DONE...












                
