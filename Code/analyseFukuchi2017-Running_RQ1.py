# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:11:38 2021

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    RQ1: Sequential analysis with increasing gait cycle number

    This analysis follows a similar line to:
        
        Taylor et al. (2015). Determining optimal trial size using sequential
        analysis. J Sports Sci, 33: 300-308
        
        Forrester (2015). Selecting the number of trials in experimental
        biomechanics studies. Int Biomech, 2: 62-72.
    
    Each discrete point of the time-normalised gait cycle is considered its own
    entity, and a moving point mean is calculated as gait cycle number increases
    across the trial. This moving point mean is compared to the mean and 0.25 SD.
    Each point will fit within this bandwidth differently, so we can look at every
    point - but for true 'stability' you probably want every point to fall within
    this bandwidth, right? This is where the 1D analysis comes in, where it looks
    at the entire curve and if it fits within the 'stability' bounds.
    
"""

# %% Import packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import _pickle as cPickle
import bz2

#Set matplotlib parameters
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# %% Define functions

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
analysisLabels = ['Hip FLEX/EXT', 'Hip ADD/ABD', 'Hip IR/ER',
                  'Knee FLEX', 'Ankle DF/PF']

#Set colour palette for analysis variables
analysisCol = sns.color_palette('colorblind', len(analysisVar))

#Set symbols for analysis variables
analysisSym = ['s', 'o', 'd', '^', 'v']

#Set trial labels for figures
trialLabels = ['2.5 m\u00b7s$^{-1}$', '3.5 m\u00b7s$^{-1}$', '4.5 m\u00b7s$^{-1}$']

#Set trial names
trialList = ['runT25','runT35','runT45']

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

#Setting for whether to run the sequential analysis or load existing
seqAnalysis = False

if seqAnalysis:

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
    
else:
    
    #Load existing data
    seqResultsDict = loadPickleDict(seqAnalysisDir+'\\Fukuchi2017-Running-seqResults.pbz2')

#Convert to dataframe
df_seqResults = pd.DataFrame.from_dict(seqResultsDict)

#Calculate mean and 95% CIs for how long each variable takes to reach 'stability'
#Other summary statistics like median, min and max values are also calculated
#This is done across both 0D and 1D variables, and different run trials

if seqAnalysis:

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
    savePickleDict(seqSummaryDict, seqAnalysisDir+'\\Fukuchi2017-Running-seqSummary.pbz2')
    
else:
    
    #Load existing dataset
    seqSummaryDict = loadPickleDict(seqAnalysisDir+'\\Fukuchi2017-Running-seqSummary.pbz2')
        
#Convert to dataframe
df_seqSummary = pd.DataFrame.from_dict(seqSummaryDict)

#Calculate proportion (%) of participants that are 'stable' at each gait cycle
#number across the different variables

#Set dictionary to store data
seqPropDict = {'trialID': [], 'analysisVar': [], 'varType': [],
               'stabilityThreshold': [], 'nGC': [], 'propStable': []}

if seqAnalysis:

    #Loop through the trial types
    for tt in range(len(trialList)):
        
        #Loop through analysis variables
        for vv in range(len(analysisVar)):
        
            #Loop through the stability levels
            for ss in range(len(stabilityLevels)):
            
                #Extract the current dataset of interest
                #OD
                df_currSeq_0D = df_seqResults.loc[(df_seqResults['trialID'] == trialList[tt]) &
                                                  (df_seqResults['analysisVar'] == analysisVar[vv]) &
                                                  (df_seqResults['stabilityThreshold'] == stabilityLevels[ss]) &
                                                  (df_seqResults['varType'] == '0D'),]
                #1D
                df_currSeq_1D = df_seqResults.loc[(df_seqResults['trialID'] == trialList[tt]) &
                                                  (df_seqResults['analysisVar'] == analysisVar[vv]) &
                                                  (df_seqResults['stabilityThreshold'] == stabilityLevels[ss]) &
                                                  (df_seqResults['varType'] == '1D'),]
                
                #Create arrays to work through a sequence of values up the max of
                #stability gait cycle numbers for this current iteration
                sequence0D = np.linspace(2,max(df_currSeq_0D['stabilityGC']),max(df_currSeq_0D['stabilityGC'])-1)
                sequence1D = np.linspace(2,max(df_currSeq_1D['stabilityGC']),max(df_currSeq_1D['stabilityGC'])-1)
                prop0D = np.zeros(len(sequence0D))
                prop1D = np.zeros(len(sequence1D))
                
                #Loop through the sequences and calculate the proportions at each 
                #gait cycle number
                #0D
                for pp in range(len(sequence0D)):
                    #Extract number of participants meeting threshold at this stage
                    #Append this to the created dictionary, along with ID data
                    seqPropDict['propStable'].append(len(np.where(df_currSeq_0D['stabilityGC'].values <= sequence0D[pp])[0].flatten()) / len(subList))
                    seqPropDict['nGC'].append(int(sequence0D[pp]))
                    seqPropDict['stabilityThreshold'].append(stabilityLevels[ss])
                    seqPropDict['varType'].append('0D')
                    seqPropDict['analysisVar'].append(analysisVar[vv])
                    seqPropDict['trialID'].append(trialList[tt])
                #1D
                for pp in range(len(sequence1D)):
                    #Extract number of participants meeting threshold at this stage
                    #Append this to the created array
                    seqPropDict['propStable'].append(len(np.where(df_currSeq_1D['stabilityGC'].values <= sequence1D[pp])[0].flatten()) / len(subList))
                    seqPropDict['nGC'].append(int(sequence1D[pp]))
                    seqPropDict['stabilityThreshold'].append(stabilityLevels[ss])
                    seqPropDict['varType'].append('1D')
                    seqPropDict['analysisVar'].append(analysisVar[vv])
                    seqPropDict['trialID'].append(trialList[tt])
                    
    #Stash the compiled sequential proportion analysis for speedier use later
    savePickleDict(seqPropDict, seqAnalysisDir+'\\Fukuchi2017-Running-seqProp.pbz2')
                     
else:
    
    #Load existing dataset
    seqPropDict = loadPickleDict(seqAnalysisDir+'\\Fukuchi2017-Running-seqProp.pbz2')

#Convert proportion dictionary to dataframe
df_seqProp = pd.DataFrame.from_dict(seqPropDict)

#Check whether to export the sequential analysis summary
if seqAnalysis:

    #Export sequential analysis results to file
    df_seqSummary.to_csv(seqAnalysisDir+'\\Fukuchi2017-Running-SequentialAnalysisSummary.csv',
                         index = False)

# %% Visualise sequential analysis results

#Replicate the box and whisker presented in Oliveira & Pirscoveanu (2021) displaying
#the number of strides to achieve stability (y-axis) and variables along the x-axis
#Split this by the hue for 0D and 1D variables

#Initialise figure
fig, ax = plt.subplots(nrows = len(trialList), ncols = len(stabilityLevels),
                       figsize = (10,8))

#Loop through trials
for tt in range(len(trialList)):
    
    #Loop through stability thresholds
    for ss in range(len(stabilityLevels)):
        
        #Extract current trial and threshold from dataframe
        df_currSeq = df_seqResults.loc[(df_seqResults['trialID'] == trialList[tt]) &
                                       (df_seqResults['stabilityThreshold'] == stabilityLevels[ss]),]

        #Plot boxplot with Seaborn
        sns.boxplot(data = df_currSeq, x = 'analysisVar', y = 'stabilityGC',
                    whis = [0,100], palette = ['k','r'], hue = 'varType', width = 0.5,
                    ax = ax[tt,ss])
        
        #Alter the faces and outlines of bars
        #Loop through boxes and fix colours
        for ii in range(len(ax[tt,ss].artists)):
        
            #Get the current artist
            artist = ax[tt,ss].artists[ii]
            
            #Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor('None')
            
            #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            #Loop over them here, and use the same colour as above
            #The first two relate to the whisker lines, so we set these to dashes
            for jj in range(ii*6,ii*6+6):
                line = ax[tt,ss].lines[jj]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
                if jj < ii*6 + 2:
                    line.set_linestyle('--')
        
        # #Add mean point on box plot
        # sns.pointplot(data = df_currSeq, x = 'analysisVar', y = 'stabilityGC',
        #               join = False, ci = 'sd',
        #               markers = ['o', 'o'], s = 5,
        #               capsize = 0.05, errwidth = 1, dodge = 0.75,
        #               palette = ['k','r'], hue = 'varType',
        #               ax = ax[tt,ss])
        
        #Mean and SD points don't really fit that well...

        #Remove legend
        ax[tt,ss].get_legend().set_visible(False)
        
        #Adjust x-axes labels
        if tt == len(trialList)-1:
            #Set x tick labels
            ax[tt,ss].xaxis.set_ticklabels(analysisLabels, rotation = 90)
            #Remove x label
            ax[tt,ss].set_xlabel('')
        else:
            #remove x-label and ticks
            ax[tt,ss].xaxis.set_ticklabels([])
            ax[tt,ss].set_xlabel('')
            
        #Set y-label and ticks
        if ss == 0:
            #Set label
            ax[tt,ss].set_ylabel('Min. No. of Gait Cycles')
            #Set y-limits
            ax[tt,ss].set_ylim([0,45])
            #Set ticks
            ax[tt,ss].set_yticks([0,10,20,30,40])
            ax[tt,ss].yaxis.set_ticklabels([0,10,20,30,40])
        else:
            #Remove label
            ax[tt,ss].set_ylabel('')
            #Set y-limits
            ax[tt,ss].set_ylim([0,45])
            #Set ticks but remove labels
            ax[tt,ss].set_yticks([0,10,20,30,40])
            ax[tt,ss].yaxis.set_ticklabels([])
        
        #Set title
        ax[tt,ss].set_title(f'{trialLabels[tt]} / {int(np.round(stabilityLevels[ss]*100))}% SD')
        
#Tight layout
plt.tight_layout()

#Save figure
fig.savefig(f'{seqAnalysisDir}\\Figures\\Boxplot_NoGaitCycles_Stability.pdf')
fig.savefig(f'{seqAnalysisDir}\\Figures\\Boxplot_NoGaitCycles_Stability.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

#Replicate the line plot presented in Oliveira & Pirscoveanu (2021) displaying
#the percentage of participants achieving stability across number of gait cycles
#Suplot this so that rows represent running speeds and columns represent 0D and 1D
#variables. The different kinematic variables will be on the x-axis and the % of
#participants achieving stability on the y-axis

#Set variable types
varTypes = ['0D', '1D'] ##### theoretically this could go earlier for looping

#Set colour palette
colourPal = sns.color_palette('viridis')

#Set marker style
markerStyle = ['o', 's', '^', 'd', 'p']

#Identify max for all sequences
maxSeqGC = max(df_seqProp['nGC'])

#Loop through stability thresholds
for ss in range(len(stabilityLevels)):

    #Initialise figure
    fig, ax = plt.subplots(nrows = len(trialList), ncols = len(varTypes),
                           figsize = (10,8))
    
    #Loop through trial list
    for tt in range(len(trialList)):
        
        #Loop through variable types
        for kk in range(len(varTypes)):
            
            #Loop through analysis variables
            for vv in range(len(analysisVar)):
            
                #Extract the current sequntial analysis dataset
                df_currSeq = df_seqProp.loc[(df_seqProp['analysisVar'] == analysisVar[vv]) &
                                            (df_seqProp['stabilityThreshold'] == stabilityLevels[ss]) &
                                            (df_seqProp['varType'] == varTypes[kk]) &
                                            (df_seqProp['trialID'] == trialList[tt]),]
                
                #Extract variables for line plot
                X = df_currSeq['nGC'].to_numpy()
                Y = df_currSeq['propStable'].to_numpy() * 100
                
                #Plot line
                ax[tt,kk].plot(X, Y, color = colourPal[vv],
                               marker = markerStyle[vv], markersize = 5,
                               label = analysisLabels[vv])
                
            #Set axes paremeters
            
            #Set y-label and ticks
            if kk == 0:
                #Set label
                ax[tt,kk].set_ylabel('% Stable Participants')
                #Set y-limits
                ax[tt,kk].set_ylim([-5,105])
                #Set ticks
                ax[tt,kk].set_yticks([0,25,50,75,100])
                ax[tt,kk].yaxis.set_ticklabels([0,25,50,75,100])
            else:
                #Remove label
                ax[tt,kk].set_ylabel('')
                #Set y-limits
                ax[tt,kk].set_ylim([-5,105])
                #Set ticks but remove labels
                ax[tt,kk].set_yticks([0,25,50,75,100])
                ax[tt,kk].yaxis.set_ticklabels([])
            
            #Adjust x-axes labels
            if tt == len(trialList)-1:
                #Set x-limits
                ax[tt,kk].set_xlim([0,maxSeqGC+1])
                #Set x-ticks
                ax[tt,kk].set_xticks([2,10,20,30,40])
                #Remove x-ticks
                ax[tt,kk].xaxis.set_ticklabels([2,10,20,30,40])
                #Set x label
                ax[tt,kk].set_xlabel('No. of Gait Cycles to Reach Stability')
            else:
                #Set x-limits
                ax[tt,kk].set_xlim([0,maxSeqGC+1])
                #Set x-ticks
                ax[tt,kk].set_xticks([2,10,20,30,40])
                #Remove x-ticks
                ax[tt,kk].xaxis.set_ticklabels([])
                #remove x-label
                ax[tt,kk].set_xlabel('')
                
            #Set legend
            if tt == len(trialList)-1 and kk == len(varTypes)-1:
                ax[tt,kk].legend()
            
            #Set title
            ax[tt,kk].set_title(f'{trialLabels[tt]} / {varTypes[kk]} Variables')
        
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(f'{seqAnalysisDir}\\Figures\\LinePlot_StableProportion_{int(np.round(stabilityLevels[ss]*100))}perSD.pdf')
    fig.savefig(f'{seqAnalysisDir}\\Figures\\LinePlot_StableProportion_{int(np.round(stabilityLevels[ss]*100))}perSD.png',
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close()
    
# %% ----- End of analyseFukuchi2017-Running_RQ1.py -----