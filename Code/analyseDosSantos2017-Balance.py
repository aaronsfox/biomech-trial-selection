# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:41:50 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function compiles the processed data from the dos Santons et al. (2017)
    balance dataset, and works through the sampling/simulation process
    of selecting data portions and comparing the outputs.
    
    Specific notes on analysis processes are outlined in the comments.
    
    TODO:
        
        > Can probably select one each of the three trials (randomly) for each
          participant to speed things up...
    
"""

# %% Import packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# %% Define functions

#COP plots
def plotCOP(subID, currSurface, currVision, df_grf):
    
    #Initialise subplot grid
    fig = plt.figure(figsize = (8,8))
    axCOP = plt.subplot2grid((4, 2), (0, 0), colspan = 2)
    axLAP = plt.subplot2grid((4, 2), (1, 0), colspan = 1)
    axLML = plt.subplot2grid((4, 2), (1, 1), colspan = 1)
    axRAP = plt.subplot2grid((4, 2), (2, 0), colspan = 1)
    axRML = plt.subplot2grid((4, 2), (2, 1), colspan = 1)
    axNAP = plt.subplot2grid((4, 2), (3, 0), colspan = 1)
    axNML = plt.subplot2grid((4, 2), (3, 1), colspan = 1)
    
    #Add titles
    fig.suptitle(subID+' '+currSurface+' '+currVision+' COP Data', fontsize=16)
    axCOP.set_title('COP Disp.')
    axLAP.set_title('LCOP Anterior/Posterior (AP)')
    axLML.set_title('LCOP Medial/Lateral (ML)')
    axRAP.set_title('RCOP Anterior/Posterior (AP)')
    axRML.set_title('RCOP Medial/Lateral (ML)')
    axNAP.set_title('NET Anterior/Posterior (AP)')
    axNML.set_title('NET Medial/Lateral (ML)')
    
    #Add axes labels
    axCOP.set_xlabel('COP ML [cm]')
    axCOP.set_ylabel('COP AP [cm]')
    axLAP.set_ylabel('LCOP [cm]')
    axRAP.set_ylabel('LCOP [cm]')
    axNAP.set_ylabel('COP NET [cm]')
    axNAP.set_xlabel('Time [s]')
    axNML.set_xlabel('Time [s]')
    
    #Set tight layout
    plt.tight_layout()
    
    #Plot data
    
    #COP position data
    axCOP.plot(df_grf['LCOP_Z'].to_numpy(),df_grf['LCOP_X'].to_numpy(),
             linestyle = '-', color = 'red', label = 'Left')
    axCOP.plot(df_grf['RCOP_Z'].to_numpy(),df_grf['RCOP_X'].to_numpy(),
             linestyle = '-', color = 'blue', label = 'Right')
    axCOP.plot(df_grf['COPNET_Z'].to_numpy(),df_grf['COPNET_X'].to_numpy(),
             linestyle = '-', color = 'black', label = 'Net')
    axCOP.legend(loc = 'best')
    
    #Left data
    axLAP.plot(df_grf['Time'].to_numpy(),df_grf['LCOP_X'].to_numpy(),
               linestyle = '-', color = 'red')
    axLAP.set_xlim([0,60])
    axLML.plot(df_grf['Time'].to_numpy(),df_grf['LCOP_Z'].to_numpy(),
               linestyle = '-', color = 'red')
    axLML.set_xlim([0,60])
    
    #Right data
    axRAP.plot(df_grf['Time'].to_numpy(),df_grf['RCOP_X'].to_numpy(),
               linestyle = '-', color = 'blue')
    axRAP.set_xlim([0,60])
    axRML.plot(df_grf['Time'].to_numpy(),df_grf['RCOP_Z'].to_numpy(),
               linestyle = '-', color = 'blue')
    axRML.set_xlim([0,60])
    
    #Net data
    axNAP.plot(df_grf['Time'].to_numpy(),df_grf['COPNET_X'].to_numpy(),
               linestyle = '-', color = 'black')
    axNAP.set_xlim([0,60])
    axNML.plot(df_grf['Time'].to_numpy(),df_grf['COPNET_Z'].to_numpy(),
               linestyle = '-', color = 'black')
    axNML.set_xlim([0,60])

# %% Set-up

#Create main directory variable
mainDir = os.getcwd()

#Create path to data storage folder
os.chdir('..\\DosSantos2017-BalanceDataset')
dataDir = os.getcwd()

#Get directory list
dList = os.listdir()
#Just keep directories with 'RBDS' for subject list
subList = []
for item in dList:
    if os.path.isdir(item):
        #Check for subject directory and append if appropriate
        if item.startswith('PDS'):
            subList.append(item)
            
#Get participant info
df_subInfo = pd.read_csv('ParticipantInfo\\PDSinfo.txt',
                         sep = '\t', header = 0)

#Convert the subject column to mtch the participant label
df_subInfo['Subject'] = df_subInfo['Subject'].apply(str) #convert to string first
for dd in range(len(df_subInfo['Subject'])):
    #Check need to add the leading zero
    if int(df_subInfo['Subject'][dd]) < 10:
        #Add trialing zero and append
        df_subInfo.at[dd, 'Subject'] = 'PDS0'+df_subInfo['Subject'][dd]
    else:
        #Leave as is and append
        df_subInfo.at[dd, 'Subject'] = 'PDS'+df_subInfo['Subject'][dd]

# %% Collate processed data

#Set dictionary to store data in
dataDict = {'subID': [], 'trialID': [], 'vision': [], 'surface': [],
            'ageGroup': [], 'gender': [], 'grf': [],
            'LCOP_AP_Disp': [], 'LCOP_ML_Disp': [], 'LCOP_Total_Disp': [],
            'RCOP_AP_Disp': [], 'RCOP_ML_Disp': [], 'RCOP_Total_Disp': [],
            'COPNET_AP_Disp': [], 'COPNET_ML_Disp': [], 'COPNET_Total_Disp': []}

#Loop through subjects
for ii in range(len(subList)):
    
    #Navigate to subject directory
    os.chdir(subList[ii])
    
    #Get trial list
    trialList = df_subInfo.loc[df_subInfo['Subject'] == subList[ii],['Trial']].values.tolist()
    #Flatten this trial list
    trialList = [item for sublist in trialList for item in sublist]
    
    #Loop through trials
    for tt in range(len(trialList)):
        
        #Load in angles and grf data for current trial
        ### Currently not using angles...
        # df_ang = pd.read_csv(trialList[tt]+'ang.txt', sep = '\t', header = 0)
        df_grf = pd.read_csv(trialList[tt]+'grf.txt', sep = '\t', header = 0)
        
        #Identify vision, surface and participant details for current trial
        df_currTrial = df_subInfo.loc[df_subInfo['Trial'] == trialList[tt],
                                      ['Vision','Surface','AgeGroup','Gender']]
        currVision = df_currTrial['Vision'].values[0]
        currSurface = df_currTrial['Surface'].values[0]
        currGroup = df_currTrial['AgeGroup'].values[0]
        currGender = df_currTrial['Gender'].values[0]
        
        #Calculate COP displacement variables for singular axes
        #Note that a zero is inserted at the start of these to align with the
        #original GRF time array
        
        #Left
        copL_AP_disp = np.concatenate(([0],np.diff(df_grf['LCOP_X'].to_numpy())), axis = 0)
        copL_ML_disp = np.concatenate(([0],np.diff(df_grf['LCOP_Z'].to_numpy())), axis = 0)
        
        #Right
        copR_AP_disp = np.concatenate(([0],np.diff(df_grf['RCOP_X'].to_numpy())), axis = 0)
        copR_ML_disp = np.concatenate(([0],np.diff(df_grf['RCOP_Z'].to_numpy())), axis = 0)
        
        #COP Net
        copNet_AP_disp = np.concatenate(([0],np.diff(df_grf['COPNET_X'].to_numpy())), axis = 0)
        copNet_ML_disp = np.concatenate(([0],np.diff(df_grf['COPNET_Z'].to_numpy())), axis = 0)
        
        #Calculate COP displacement variables for total values
        #Note that a zero is inserted at the start of these to align with the
        #original GRF time array
        
        #Left
        copL_totalDisp = np.zeros(len(df_grf))
        for pp in range(1,len(df_grf)):
            #Get data points
            x1 = df_grf['LCOP_X'].to_numpy()[pp-1]
            y1 = df_grf['LCOP_Z'].to_numpy()[pp-1]
            x2 = df_grf['LCOP_X'].to_numpy()[pp]
            y2 = df_grf['LCOP_Z'].to_numpy()[pp]
            #Calculate distance and append to array
            copL_totalDisp[pp] = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
            
        #Right
        copR_totalDisp = np.zeros(len(df_grf))
        for pp in range(1,len(df_grf)):
            #Get data points
            x1 = df_grf['RCOP_X'].to_numpy()[pp-1]
            y1 = df_grf['RCOP_Z'].to_numpy()[pp-1]
            x2 = df_grf['RCOP_X'].to_numpy()[pp]
            y2 = df_grf['RCOP_Z'].to_numpy()[pp]
            #Calculate distance and append to array
            copR_totalDisp[pp] = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
        
        #Net
        copNet_totalDisp = np.zeros(len(df_grf))
        for pp in range(1,len(df_grf)):
            #Get data points
            x1 = df_grf['COPNET_X'].to_numpy()[pp-1]
            y1 = df_grf['COPNET_Z'].to_numpy()[pp-1]
            x2 = df_grf['COPNET_X'].to_numpy()[pp]
            y2 = df_grf['COPNET_Z'].to_numpy()[pp]
            #Calculate distance and append to array
            copNet_totalDisp[pp] = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
            
        
        #Append data to dictionary
        dataDict['subID'].append(subList[ii])
        dataDict['trialID'].append(trialList[tt])
        dataDict['vision'].append(currVision)
        dataDict['surface'].append(currSurface)
        dataDict['ageGroup'].append(currGroup)
        dataDict['gender'].append(currGender)
        dataDict['grf'].append(df_grf)
        dataDict['LCOP_AP_Disp'].append(copL_AP_disp)
        dataDict['LCOP_ML_Disp'].append(copL_ML_disp)
        dataDict['LCOP_Total_Disp'].append(copL_totalDisp)
        dataDict['RCOP_AP_Disp'].append(copR_AP_disp)
        dataDict['RCOP_ML_Disp'].append(copR_ML_disp)
        dataDict['RCOP_Total_Disp'].append(copR_totalDisp)
        dataDict['COPNET_AP_Disp'].append(copNet_AP_disp)
        dataDict['COPNET_ML_Disp'].append(copNet_ML_disp)
        dataDict['COPNET_Total_Disp'].append(copNet_totalDisp)
        
        #Plot data for later error checking
        plotCOP(subList[ii], currSurface, currVision, df_grf)
        
        #Save figure
        plt.savefig(trialList[tt]+'_copFig.png')
        plt.close()
        
    #Print confirmation
    print('Data extracted for '+subList[ii]+'. '+str(ii+1)+' of '+str(len(subList))+' done...')
        
    #Return to data directory
    os.chdir(dataDir)
    
#Convert dictionary to dataframe
df_data = pd.DataFrame.from_dict(dataDict)

#### TODO: Loop through subjects and randomly select one of each trial type
#### for further analysis...
    
# %% Run tests

#Settings for subsequent tests

# #Set analysis variable
# analysisVar = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
#                'knee_angle_r', 'ankle_angle_r']

# #Set labels for analysis variables
# analysisLabels = ['Hip Flexion', 'Hip Adduction', 'Hip Rotation',
#                   'Knee Flexion', 'Ankle Dorsi/Plantarflexion']

# #Set colour palette for analsis variables
# analysisCol = sns.color_palette('colorblind', len(analysisVar))

# #Set symbols for analysis variables
# analysisSym = ['s', 'o', 'd', '^', 'v']

#Set the alpha level for the t-tests
alpha = 0.05

#Set duration splits to work through
durSplits = np.linspace(5, 60, int((60-5)/5)+1)

#Set vision condition variable
visionList = ['Open', 'Closed']

#Set surface condition list
surfaceList = ['Rigid', 'Foam']

#Set age group condition list
groupList = ['Young', 'Old']

# %% Sequential analysis with increasing analysis duration

##### TODO
    # Split this across the various grouping conditions

# This analysis follows a similar line to:
    
    # Taylor et al. (2015). Determining optimal trial size using sequential
    # analysis. J Sports Sci, 33: 300-308
    #
    # Forrester (2015). Selecting the number of trials in experimental
    # biomechanics studies. Int Biomech, 2: 62-72.

# Variables are calculated based on the extracted set of data in one second
# intervals, and a moving point mean is therefore calculated. This moving point
# mean is compared to the mean and a proportion of the SD (0.25 is used in the
# above references for example). 'Stability' is considered when the moving point
# mean falls within the 'bandwidth' and continues this way.

#Set a dictionary to store findings of sequential analysis
#### TODO: Add spot for the data???
seqDict = {'duration': [], 'subID': [], 'trialID': [],
           'vision': [], 'surface': [], 'ageGroup': [], 'gender': [],
           'analysisVar': [], 'seqVal': []}

#Loop through trials
for tt in range(len(df_data)):
    
    #Get current trial label
    currTrial = df_data['trialID'][tt]
    
    #Extract the dataframe for the current trial ID
    df_currTrial = df_data.loc[df_data['trialID'] == currTrial,].reset_index(drop = True)

    #### TODO
        # There is space within this framework to do multiple variables. For now
        # we'll start with mean COP displacement
        
    #Calculate the total mean and standard deviation using the entire dataset
    time = df_currTrial['grf'][0]['Time'].to_numpy()
    copTotalDisp = df_currTrial['COPNET_Total_Disp'].to_numpy()[0]
    total_m = np.mean(copTotalDisp, axis = 0)
    total_sd = np.std(copTotalDisp, axis = 0)
    
    #Loop through 5 second periods and calculate mean and standard deviation
    for dd in range(len(durSplits)):
        
        #Set end time
        #Data ends at 59.99, so if on last iteration need to come back
        if durSplits[dd] == 60.0:
            endTime = durSplits[dd] - 0.01
        else:
            endTime = durSplits[dd]
            
        #Identify indices for start and end time
        endInd = np.where(time == endTime)[0][0]
        
        #Extract the data for the desired period
        currData = copTotalDisp[0:endInd]
        
        #Calculate mean and SD for current number of cycles
        curr_m = np.mean(currData, axis = 0)
        
        #Normalise to zero mean and 1 SD
        curr_norm = (curr_m - total_m) / total_sd
        
        #Add data to dictionary
        seqDict['duration'].append(durSplits[dd])
        seqDict['subID'].append(df_currTrial['subID'][0])
        seqDict['trialID'].append(df_currTrial['trialID'][0])
        seqDict['vision'].append(df_currTrial['vision'][0])
        seqDict['surface'].append(df_currTrial['surface'][0])
        seqDict['ageGroup'].append(df_currTrial['ageGroup'][0])
        seqDict['gender'].append(df_currTrial['gender'][0])
        ####
        #### TODO: fix with different variables...
        ####
        seqDict['analysisVar'].append('COPNET_Total_Disp')
        seqDict['seqVal'].append(curr_norm)

#Convert dictionary to a dataframe
df_seqAnalysis = pd.DataFrame.from_dict(seqDict)


#Sample box plot of data for COP total displacement

#####
#####
##### TODO: FIX UP!
#####
#####
##### NOTE: there are multiple options for segregating data in this dataset
#####
#####

#Initialise figure
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,3.5))

#Plot the 0.25 SD bandwidth
ax.axhline(y = 0.25, linewidth = 1, linestyle = '--', color = 'grey')
ax.axhline(y = -0.25, linewidth = 1, linestyle = '--', color = 'grey')

#Extract current analysis variable dataframe
df_currSeq = df_seqAnalysis.loc[(df_seqAnalysis['analysisVar'] == 'COPNET_Total_Disp'),]

##### NOTE: below plot doesn't consider multiple trials by certain subjects

#Plot boxplot with Seaborn
sns.boxplot(data = df_currSeq, x = 'duration', y = 'seqVal', hue = 'vision',
            whis = [0,100], palette = 'colorblind', ax = ax)

#### Boxplots demonstrate the duration for *EVERYONE* to get under
#### the 0.25 SD threshold --- but this could actually
#### vary from person to person...
####
#### Should calculate this and for each variable calculate the duration
#### it takes to get to their stability point...
####
#### Also consider the appropriateness of a 0.25 SD threshold with the absolute
#### maximum value here --- it could be quite sensitive and may be valid to calculate
#### the duration for variable thresholds...

# %% Determine period extraction points

### TODO:
    #We could actually go higher for comparison to 'ground truth' given we don't
    #need to compare two sets of durations cycles...

#Set the list of gait cycles to extract
minExtract = 5
maxExtract = 25
extractInterval = 5
extractDur = np.linspace(minExtract,maxExtract,extractInterval)

#Set the number of sampling iterations to run
nSamples = 1000

#Set dictionary to store gait cycle points in
sampleDict = {'subID': [], 'trialID': [],
              'vision': [], 'surface': [], 'ageGroup': [], 'gender': [],
              'extractDuration': [], 'startPoint1': [], 'startPoint2': []}

#Loop through trial list
for tt in range(len(df_data)):
       
    #Loop through extraction numbers
    for ee in range(len(extractDur)):
        
        #Determine the 'go zones' for random selection of the first period
        #Round seconds are noted here as potential start points
        #All trials go for 1 minute (i.e. 60 seconds)
        durStarts = np.linspace(0,60,60+1)
        #Remove the final X values relative to current extraction duration, as
        #these won't leave enough to grab
        durStarts = durStarts[0:int(-extractDur[ee])]
        #Loop through the start numbers and check if using it there will be
        #a valid number before or after it
        goZone = list()
        for gg in range(len(durStarts)):
            #Check whether there will be valid values after
            enoughAfter = durStarts[-1] - (durStarts[gg] + extractDur[ee]) > 0
            #Check whether there would be valid values before
            enoughBefore = durStarts[gg] - extractDur[ee] > 0
            #If one of these is True then the value can be added to the 'go zone'
            if enoughAfter or enoughBefore:
                goZone.append(durStarts[gg])
                
        #Create list to store starting gait values into for extraction
        start1 = list()
        start2 = list()
        
        #Set seed here for sampling consistency
        random.seed(12345)
        
        #Loop through sampling number
        for ss in range(nSamples):
            
            #Append subject and trial ID names
            sampleDict['subID'].append(df_data['subID'][tt])
            sampleDict['trialID'].append(df_data['trialID'][tt])
            sampleDict['vision'].append(df_data['vision'][tt])
            sampleDict['surface'].append(df_data['surface'][tt])
            sampleDict['ageGroup'].append(df_data['ageGroup'][tt])
            sampleDict['gender'].append(df_data['gender'][tt])
            
            #Append extract number details
            sampleDict['extractDuration'].append(int(extractDur[ee]))
            
            #Select a random number from the list to start from
            s1 = random.choice(goZone)
            
            #Set a list to make second selection from
            select2 = list()
            
            #At this point split into two lists so length checks are easier
            #Can't use preceding starting points if they are within the
            #extraction number of the starting point
            goZone1 = [x for x in goZone if x < s1-extractDur[ee]+1]
            #Can't use values that will be encompassed within the gait cycles
            #extracted from the first starting point
            goZone2 = [x for x in goZone if x > s1+extractDur[ee]-1]
            #Concatenate the lists for to select from
            select2 = goZone1 + goZone2

            #Select a random number from the second list to start from
            s2 = random.choice(select2)
            
            #Set strikes for current sample in dictionary
            sampleDict['startPoint1'].append(s1)
            sampleDict['startPoint2'].append(s2)

#Convert dictionary to a dataframe
df_samples = pd.DataFrame.from_dict(sampleDict)

# %% Duration comparison to 'ground truth' mean
