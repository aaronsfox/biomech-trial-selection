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
import spm1d

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

# %% Set-up

#Create main directory variable
mainDir = os.getcwd()

#Create path to data storage folder
os.chdir('..\\Fukuchi2017-RunningDataset')
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

# %% Collate processed data

#Set trial names to extract
#Each participant at least has the 25, 35 and 45 runs (some have 30)
trialList = ['runT25','runT35','runT45']

#Set dictionary to store data in
dataDict = {'subID': [], 'trialID': [], 'kinematics': [],
            'rightFS': [], 'leftFS': [], 
            'rGC_n': [], 'lGC_n': []}

#Loop through subjects
for ii in range(len(subList)):
# for ii in range(0,20):
    
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
        
        #Plot data for later error checking
        plotKinematics(subList[ii], trialList[tt], df_ik, leftFS, rightFS)
        
        #Save figure
        plt.savefig(subList[ii]+'_'+trialList[tt]+'_kinematics.png')
        plt.close()
        
    #Print confirmation
    print('Data extracted for '+subList[ii])
        
    #Return to data directory
    os.chdir(dataDir)
    
# %% Determine minimum gait cycles

#This analysis will focus on right limb data, so we'll focus on these gait cycle n's
minGC = min(dataDict['rGC_n'])

### NOTE
    #The above calculation determines 34 to be the minimum number of gait cycles
    #across all *CURRENT* participants
    #Considering this we can probably safely go up to 15 and have minimum overlap

### TODO
    #Update and check once final participants added
    
# %% TODO: Determine most and least variable...

#Likely knee flexion and hip rotation...
    
# %% Determine gait cycle extraction points

#Set the list of gait cycles to extract
minExtract = 5
maxExtract = 15
extractNo = np.linspace(minExtract,maxExtract,
                        maxExtract-minExtract+1)

#Set the number of sampling iterations to run
nSamples = 1000

#Set dictionary to store gait cycle points in
sampleDict = {'subID': [], 'trialID': [],
              'extractNo': [], 'footStrikes1': [], 'footStrikes2': []}

#Loop through subjects
for ii in range(len(subList)):
# for ii in range(0,20):
    
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
            
            #Determine the 'go zones' for random selection of the first set
            #Start by creating each individual cycle in a list
            gcStarts = np.linspace(0,nGC,nGC+1)
            #Remove the final X values relative to current extraction number, as
            #these won't leave enough to grab
            gcStarts = gcStarts[0:int(-extractNo[ee])]
            #Loop through the start numbers and check if using it there will be
            #a valid number before or after it
            goZone = list()
            for gg in range(len(gcStarts)):
                #Check whether there will be valid values after
                enoughAfter = gcStarts[-1] - (gcStarts[gg] + extractNo[ee]) > 0
                #Check whether there would be valid values before
                enoughBefore = gcStarts[gg] - extractNo[ee] > 0
                #If one of these is True then the value can be added to the 'go zone'
                if enoughAfter or enoughBefore:
                    goZone.append(gcStarts[gg])
                    
            #Create list to store starting gait values into for extraction
            start1 = list()
            start2 = list()
            
            #Set seed here for sampling consistency
            random.seed(12345)
            
            #Loop through sampling number
            for ss in range(nSamples):
                
                #Append subject and trial ID names
                sampleDict['subID'].append(subList[ii])
                sampleDict['trialID'].append(trialList[tt])
                
                #Append extract number details
                sampleDict['extractNo'].append(int(extractNo[ee]))
                
                #Select a random number from the list to start from
                s1 = random.choice(goZone)
                
                #Set a list to make second selection from
                select2 = list()
                
                #At this point split into two lists so length checks are easier
                #Can't use preceding starting points if they are within the
                #extraction number of the starting point
                goZone1 = [x for x in goZone if x < s1-extractNo[ee]+1]
                #Can't use values that will be encompassed within the gait cycles
                #extracted from the first starting point
                goZone2 = [x for x in goZone if x > s1+extractNo[ee]-1]
                #Concatenate the lists for to select from
                select2 = goZone1 + goZone2

                #Select a random number from the second list to start from
                s2 = random.choice(select2)
                
                #Set strikes for current sample in dictionary
                sampleDict['footStrikes1'].append(list(map(round,list(np.linspace(s1,s1+int(extractNo[ee]),int(extractNo[ee]+1))))))
                sampleDict['footStrikes2'].append(list(map(round,list(np.linspace(s2,s2+int(extractNo[ee]),int(extractNo[ee]+1))))))

#Convert dictionary to a dataframe
df_samples = pd.DataFrame.from_dict(sampleDict)
                
# %% Extract data and run tests

#Set a dictionary to store findings of each iteration in
##### TODO
    #Include some sort of standardised error/effect of difference???
resultsDict = {'extractNo': [], 'trialID': [],
               'rejectH0': [], 'pVal': [],
               'analysisVar': [],
               'Y1': [], 'Y2': [],
               'Y1m': [], 'Y2m': [], 'meanAbsError': []}
    
##### TODO:
    #Selecting variables --- start by testing knee flexion

#Set analysis variable
analysisVar = 'knee_angle_r'
    
#Set the alpha level for the t-tests
alpha = 0.05

#Loop through the different trial types
for tt in range(len(trialList)):
    
    #Extract the dataframe for the current trial ID
    df_currTrial = df_samples.loc[df_samples['trialID'] == trialList[tt],]

    #Loop through the extraction numbers
    for ee in range(len(extractNo)):
        
        #Set current extract number
        currNo = int(extractNo[ee])
        
        #Extract the dataframe for the current extraction number
        df_currExtract = df_currTrial.loc[df_currTrial['extractNo'] == currNo,]
        
        #Loop through the sampling number
        for ss in range(nSamples):
            
            #Set array to store each subjects two datasets for this sample iteration
            # Y1 = np.empty((len(subList),101))
            # Y2 = np.empty((len(subList),101))
            Y1 = np.empty((20,101))
            Y2 = np.empty((20,101))
            
            #Loop through subjects and get their data for the current sample
            # for ii in range(len(subList)):
            for ii in range(0,20):
                
                #Extract the dataframe that matches the current subject
                #As part of this also extract the row that matches the current
                #sampling iteration index
                df_currSub = df_currExtract.loc[df_currExtract['subID'] == subList[ii],]
                df_currIter = df_currSub.iloc[ss]
                
                #Set a space to store normalised data into for calculating
                #current subjects mean
                normData1 = np.empty((currNo,101))
                normData2 = np.empty((currNo,101))
                
                #Extract the current participants kinematic data relevant to
                #current trial tupe. Get the index corresponding to this in the
                #data dictionary.
                subInd = [pp for pp, bb in enumerate(dataDict['subID']) if bb == subList[ii]]
                trialInd = [kk for kk, bb in enumerate(dataDict['trialID']) if bb == trialList[tt]]
                currInd = list(set(subInd) & set(trialInd))[0]
                
                #Get the right foot strike indices
                rightFS = dataDict['rightFS'][currInd]
                
                #Get the IK dataframe
                df_ik = dataDict['kinematics'][currInd]
                
                #Loop through extraction number and normalise kinematics
                for nn in range(currNo):
                    
                    #Extract data between current heel strikes
                
                    #Get start and end time
                    startTime1 = rightFS[df_currIter['footStrikes1'][nn]]
                    endTime1 = rightFS[df_currIter['footStrikes1'][nn+1]]
                    startTime2 = rightFS[df_currIter['footStrikes2'][nn]]
                    endTime2 = rightFS[df_currIter['footStrikes2'][nn+1]]
                    
                    #Create a boolean mask for in between event times
                    extractTime1 = ((df_ik['time'] > startTime1) & (df_ik['time'] < endTime1)).values
                    extractTime2 = ((df_ik['time'] > startTime2) & (df_ik['time'] < endTime2)).values
                    
                    #Extract the time values
                    timeVals1 = df_ik['time'][extractTime1].values
                    timeVals2 = df_ik['time'][extractTime2].values
                    
                    #Extract the data
                    dataVals1 = df_ik[analysisVar][extractTime1].values
                    dataVals2 = df_ik[analysisVar][extractTime2].values
                    
                    #Normalise data to 0-100%
                    newTime1 = np.linspace(timeVals1[0],timeVals1[-1],101)
                    newTime2 = np.linspace(timeVals2[0],timeVals2[-1],101)
                    interpData1 = np.interp(newTime1,timeVals1,dataVals1)                    
                    interpData2 = np.interp(newTime2,timeVals2,dataVals2)
                    
                    #Store interpolated data in array
                    normData1[nn,:] = interpData1
                    normData2[nn,:] = interpData2
            
                #Calculate the mean of the current subjects normalised data
                #Store in the current sample iterations array for SPM1D analysis
                Y1[ii,:] = np.mean(normData1, axis = 0)
                Y2[ii,:] = np.mean(normData2, axis = 0)
            
            #Conduct the SPM1D t-test on this sample
            t = spm1d.stats.ttest_paired(Y1, Y2)
            ti = t.inference(alpha, two_tailed = True, interp = True)
            
            # #Visuliase
            # #Set-up plot
            # plt.figure(figsize=(8, 3.5))
            # #Plot mean and SD of two samples
            # ax1 = plt.axes((0.1, 0.15, 0.35, 0.8))
            # spm1d.plot.plot_mean_sd(Y1, linecolor = 'b', facecolor = 'b')
            # spm1d.plot.plot_mean_sd(Y1, linecolor = 'r', facecolor='r')
            # ax1.axhline(y = 0, color = 'k', linestyle=':')
            # ax1.set_xlabel('0-100% Gait Cycle')
            # ax1.set_ylabel(analysisVar)
            # #Plot SPM results
            # ax2 = plt.axes((0.55,0.15,0.35,0.8))
            # ti.plot()
            # ti.plot_threshold_label(fontsize = 8)
            # ti.plot_p_values(size = 10, offsets = [(0,0.3)])
            # ax2.set_xlabel('0-100% Gait Cycle')
            # #Show plot
            # plt.show()
            
            #Calculate mean absolute error of current two curves
            Y1m = np.mean(Y1, axis = 0)
            Y2m = np.mean(Y2, axis = 0)
            mae = np.mean(abs(Y1m - Y2m))            
            
            #Collate results from this sampling iteration into dictionary
            resultsDict['extractNo'].append(currNo)
            resultsDict['trialID'].append(trialList[tt])
            resultsDict['rejectH0'].append(ti.h0reject)
            resultsDict['pVal'].append(ti.p) #note there are no p-values for non-statistically significant results
            resultsDict['analysisVar'].append(analysisVar)
            resultsDict['Y1'].append(Y1)
            resultsDict['Y2'].append(Y2)
            resultsDict['Y1m'].append(Y1m)
            resultsDict['Y2m'].append(Y2m)
            resultsDict['meanAbsError'].append(mae)
            
            #Print confirmation
            print('Completed '+str(ss+1)+' of '+str(nSamples)+' for '+
                  str(currNo)+' gait cycles of '+trialList[tt])
                
# %% Sequential analysis:
    
    #See:
        
        #Taylor et al. (2015). Determining optimal trial size using sequential analysis
        #Forrester (2015). Selecting the number of trials in experimental biomechanics studies
        #Severin et al. (2019). The required number of trials for biomechanical analysis of a golf swing
        
        #Check these papers reference lists as well...