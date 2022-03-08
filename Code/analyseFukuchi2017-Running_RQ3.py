# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:25:47 2021

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    RQ3: Comparing samples from different sections of trial

    This analysis compares a sample of an individuals treadmill bout using a certain
    number of gait cycles to create the mean, to another section of the treadmill
    bout using the same number of gait cycles. Effectively this question answers
    how much of an effect using a certain subset of gait cycles from a part of
    a continuous treadmill bout has on the representative mean obtained.
    
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

#Samples comparison directory
os.chdir('..\\SamplesComp')
samplesCompDir = os.getcwd()

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

# %% Collate comparisons between sampling

#Set the list of gait cycles to extract
minExtract = 5
maxExtractSingle = 30
maxExtractDual = 15
extractNo = np.linspace(minExtract,maxExtractSingle,
                        maxExtractSingle-minExtract+1)
extractNoDual = np.linspace(minExtract,maxExtractDual,
                            maxExtractDual-minExtract+1)

#Set the number of sampling iterations
nSamples = 1000

#Load the existing samples comparison dictionaries and convert to dataframe format

#Loop through trial types
for tt in range(len(trialList)):

    #Loop through analysis variables
    for vv in range(len(analysisVar)):

        #Loop through extraction numbers
        for ee in range(len(extractNoDual)):
            
            #Set current extraction number
            currNo = int(extractNoDual[ee])
        
            #Load data dictionary for current extraction number
            samplesCompDict = loadPickleDict(samplesCompDir+'\\Fukuchi2017-Running-samplesComp-n'+
                                             str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'-'+
                                             trialList[tt]+'.pbz2')
            
            #Convert to dataframe.
            #If first iteration, set as the overall dataframe
            if tt == 0 and vv == 0 and ee == 0: 
                df_samplesComp =  pd.DataFrame.from_dict(samplesCompDict)
            else:
                #Convert to generic dataframe variable and bind to remaining
                df = pd.DataFrame.from_dict(samplesCompDict)
                df_samplesComp = pd.concat([df_samplesComp,df])
                
# %% Visualise samples comparisons and effect magnitudes

##### TODO: should tabulate these in some way for trial to trial comparison?

##### Heat map for error compared to mean...?

#Set colours to distinguish the similar looking plots for speed
colourMap = ['#4885ed', '#fdb61c', '#e54078']
colourMap2 = ['#174492', '#fd7e1c', '#a91a4a']

#0D

#Display mean absolute errors for 0D variables compared to ground truth as boxplots

#Loop through trials
for tt in range(len(trialList)):

    #Set up figure
    fig, ax = plt.subplots(nrows = 3, ncols = 2,
                            figsize = (10,8))
    
    #Set variable for axes to plot on
    whichAx = [[0,0], [0,1],
               [1,0], [1,1],
               [2,0], [2,1]]
    
    #Loop through variables to plot
    for vv in range(len(analysisVar)):
        
        #Extract current variable data for 0D variable
        df_currData = df_samplesComp.loc[(df_samplesComp['analysisVar'] == analysisVar[vv]) &
                                         (df_samplesComp['trialID'] == trialList[tt]) &
                                         (df_samplesComp['varType'] == '0D'),
                                         ['extractNo','trialID','analysisVar','meanAbsError','effectSize']]
        
        #Plot on current axes
        sns.boxplot(data = df_currData, x = 'extractNo', y = 'meanAbsError',
                    whis = [0,100], color = colourMap[tt], width = 0.75,
                    zorder = 5, ax = ax[whichAx[vv][0],whichAx[vv][1]])
        
        #Alter the faces and outlines of bars
        #Loop through boxes and fix colours
        for ii in range(len(ax[whichAx[vv][0],whichAx[vv][1]].artists)):
        
            #Get the current artist
            artist = ax[whichAx[vv][0],whichAx[vv][1]].artists[ii]
            
            #Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor('None')
            
            #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            #Loop over them here, and use the same colour as above
            #The first two relate to the whisker lines, so we set these to dashes
            for jj in range(ii*6,ii*6+6):
                line = ax[whichAx[vv][0],whichAx[vv][1]].lines[jj]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
                if jj < ii*6 + 2:
                    line.set_linestyle('--')
                    
        #Add violin plot
        sns.violinplot(data = df_currData, x = 'extractNo', y = 'meanAbsError',
                       cut = True, scale = 'width', inner = None,
                       color = colourMap[tt], width = 0.75,
                       zorder = 4, ax = ax[whichAx[vv][0],whichAx[vv][1]])
        
        #Adjust alpha and edge width on violins
        for violin in ax[whichAx[vv][0],whichAx[vv][1]].collections:
            violin.set_alpha(0.3)
            violin.set_linewidth(0)
            
        #Add point plot with mean and standard deviation
        sns.pointplot(data = df_currData, x = 'extractNo', y = 'meanAbsError', ci = 'sd',
                      join = False, color = colourMap2[tt],
                      scale = 0.75, errwidth = 2,
                      zorder = 5, ax = ax[whichAx[vv][0],whichAx[vv][1]])

        #Adjust x-axes labels
        if whichAx[vv] == [2,0] or whichAx[vv] == [1,1]:
            #Set x-limits
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[0]-0.5,
                                                        ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[1]+0.5])
            #Set x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/5)+1))
            #Set label
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlabel('No. of Gait Cycles')
        else:
            #Set x-limits
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[0]-0.5,
                                                        ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[1]+0.5])
            #Set x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/5)+1))
            #Remove x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].xaxis.set_ticklabels([])
            #Remove label
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlabel('')
            
        #Set y-label
        if whichAx[vv][1] == 0:
            ax[whichAx[vv][0],whichAx[vv][1]].set_ylabel('Absolute Difference (\u00b0)')
        else:
            ax[whichAx[vv][0],whichAx[vv][1]].set_ylabel('')
        
        #Set title
        ax[whichAx[vv][0],whichAx[vv][1]].set_title(f'Peak {analysisLabels[vv]} at {trialLabels[tt]}')
            
    #Set final axis to invisible
    ax[2,1].set_visible(False)
    
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_{trialList[tt]}_0D.pdf')
    fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_{trialList[tt]}_0D.png',
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close() 

##### TODO: should tabulate these in some way for ground truth comp as there
##### are multiple means for each extraction number???    
    
#1D

#Display peak absolute errors for 1D variables compared to ground truth as boxplots

#Loop through trials
for tt in range(len(trialList)):

    #Set up figure
    fig, ax = plt.subplots(nrows = 3, ncols = 2,
                           figsize = (10,8))
    
    #Set variable for axes to plot on
    whichAx = [[0,0], [0,1],
               [1,0], [1,1],
               [2,0], [2,1]]
    
    #Loop through variables to plot
    for vv in range(len(analysisVar)):
        
        #Extract current variable data for 0D variable
        df_currData = df_samplesComp.loc[(df_samplesComp['analysisVar'] == analysisVar[vv]) &
                                         (df_samplesComp['trialID'] == trialList[tt]) &
                                         (df_samplesComp['varType'] == '1D'),
                                         ['extractNo','trialID','analysisVar','peakAbsError','effectSize']]
        
        #Plot on current axes
        sns.boxplot(data = df_currData, x = 'extractNo', y = 'peakAbsError',
                    whis = [0,100], color = colourMap[tt], width = 0.75,
                    ax = ax[whichAx[vv][0],whichAx[vv][1]])
        
        #Alter the faces and outlines of bars
        #Loop through boxes and fix colours
        for ii in range(len(ax[whichAx[vv][0],whichAx[vv][1]].artists)):
        
            #Get the current artist
            artist = ax[whichAx[vv][0],whichAx[vv][1]].artists[ii]
            
            #Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor('None')
            
            #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            #Loop over them here, and use the same colour as above
            #The first two relate to the whisker lines, so we set these to dashes
            for jj in range(ii*6,ii*6+6):
                line = ax[whichAx[vv][0],whichAx[vv][1]].lines[jj]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
                if jj < ii*6 + 2:
                    line.set_linestyle('--')
                    
        #Add violin plot
        sns.violinplot(data = df_currData, x = 'extractNo', y = 'peakAbsError',
                       cut = True, scale = 'width', inner = None,
                       color = colourMap[tt], width = 0.75,
                       zorder = 4, ax = ax[whichAx[vv][0],whichAx[vv][1]])
        
        #Adjust alpha and edge width on violins
        for violin in ax[whichAx[vv][0],whichAx[vv][1]].collections:
            violin.set_alpha(0.3)
            violin.set_linewidth(0)
            
        #Add point plot with mean and standard deviation
        sns.pointplot(data = df_currData, x = 'extractNo', y = 'peakAbsError', ci = 'sd',
                      join = False, color = colourMap2[tt],
                      scale = 0.75, errwidth = 2,
                      zorder = 5, ax = ax[whichAx[vv][0],whichAx[vv][1]])

        #Adjust x-axes labels
        if whichAx[vv] == [2,0] or whichAx[vv] == [1,1]:
            #Set x-limits
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[0]-0.5,
                                                        ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[1]+0.5])
            #Set x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/5)+1))
            #Set label
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlabel('No. of Gait Cycles')
        else:
            #Set x-limits
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlim([ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[0]-0.5,
                                                        ax[whichAx[vv][0],whichAx[vv][1]].get_xlim()[1]+0.5])
            #Set x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/5)+1))
            #Remove x-ticks
            ax[whichAx[vv][0],whichAx[vv][1]].xaxis.set_ticklabels([])
            #Remove label
            ax[whichAx[vv][0],whichAx[vv][1]].set_xlabel('')
            
        #Set y-label
        if whichAx[vv][1] == 0:
            ax[whichAx[vv][0],whichAx[vv][1]].set_ylabel('Peak Absolute Difference (\u00b0)')
        else:
            ax[whichAx[vv][0],whichAx[vv][1]].set_ylabel('')
        
        #Set title
        ax[whichAx[vv][0],whichAx[vv][1]].set_title(f'1D {analysisLabels[vv]} at {trialLabels[tt]}')
            
    #Set final axis to invisible
    ax[2,1].set_visible(False)
    
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_{trialList[tt]}_1D.pdf')
    fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_{trialList[tt]}_1D.png',
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close()
    
# %% Visualise samples comparisons and effect magnitudes (combined speeds)

#0D

#Display mean absolute errors for 0D variables compared to ground truth as boxplots

#Set up figure
fig, ax = plt.subplots(nrows = 5, ncols = 1,
                        figsize = (8.27,11.69))

#Loop through variables to plot
for vv in range(len(analysisVar)):
    
    #Extract current variable data for 0D variable
    df_currData = df_samplesComp.loc[(df_samplesComp['analysisVar'] == analysisVar[vv]) &
                                     (df_samplesComp['varType'] == '0D'),
                                     ['extractNo','trialID','analysisVar','meanAbsError','effectSize']]
    
    #Plot on current axes
    sns.boxplot(data = df_currData,
                x = 'extractNo', y = 'meanAbsError', hue = 'trialID',
                whis = [0,100], palette = colourMap, width = 0.75,
                zorder = 5, ax = ax[vv])
    
    #Alter the faces and outlines of bars
    #Loop through boxes and fix colours
    for ii in range(len(ax[vv].artists)):
    
        #Get the current artist
        artist = ax[vv].artists[ii]
        
        #Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        artist.set_edgecolor(col)
        artist.set_facecolor('None')
        
        #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        #Loop over them here, and use the same colour as above
        #The first two relate to the whisker lines, so we set these to dashes
        for jj in range(ii*6,ii*6+6):
            line = ax[vv].lines[jj]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            if jj < ii*6 + 2:
                line.set_linestyle('--')
                
    #Add violin plot
    sns.violinplot(data = df_currData,
                   x = 'extractNo', y = 'meanAbsError', hue = 'trialID',
                   cut = True, scale = 'width', inner = None,
                   palette = colourMap, width = 0.75,
                   zorder = 4, ax = ax[vv])
    
    #Adjust alpha and edge width on violins
    for violin in ax[vv].collections:
        violin.set_alpha(0.3)
        violin.set_linewidth(0)
        
    #Add point plot with mean and standard deviation
    sns.pointplot(data = df_currData,
                  x = 'extractNo', y = 'meanAbsError', hue = 'trialID', ci = 'sd',
                  join = False, palette = colourMap2,
                  scale = 0.6, errwidth = 1.5, dodge = 0.5,
                  zorder = 5, ax = ax[vv])

    #Adjust x-axes labels
    if vv == len(analysisVar)-1:
        #Set x-limits
        ax[vv].set_xlim([ax[vv].get_xlim()[0]-0.5, ax[vv].get_xlim()[1]+0.5])
        #Set x-ticks
        ax[vv].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/1)+1))
        #Set label
        ax[vv].set_xlabel('No. of Gait Cycles')
    else:
        #Set x-limits
        ax[vv].set_xlim([ax[vv].get_xlim()[0]-0.5, ax[vv].get_xlim()[1]+0.5])
        #Set x-ticks
        ax[vv].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/1)+1))
        #Remove x-ticks
        ax[vv].xaxis.set_ticklabels([])
        #Remove label
        ax[vv].set_xlabel('')
        
    #Turn off legend
    ax[vv].get_legend().remove()
        
    #Set y-label
    ax[vv].set_ylabel('Abs. Difference (\u00b0)')
    
    #Set title
    ax[vv].set_title(f'Peak {analysisLabels[vv]}')
        
#Tight layout
plt.tight_layout()

#Save figure
fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_ALL_0D.pdf')
fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_ALL_0D.png',
            format = 'png', dpi = 300)

#Close figure
plt.close() 
    
#1D

#Set up figure
fig, ax = plt.subplots(nrows = 5, ncols = 1,
                        figsize = (8.27,11.69))

#Loop through variables to plot
for vv in range(len(analysisVar)):
    
    #Extract current variable data for 0D variable
    df_currData = df_samplesComp.loc[(df_samplesComp['analysisVar'] == analysisVar[vv]) &
                                     (df_samplesComp['varType'] == '1D'),
                                     ['extractNo','trialID','analysisVar','peakAbsError','effectSize']]
    
    #Plot on current axes
    sns.boxplot(data = df_currData,
                x = 'extractNo', y = 'peakAbsError', hue = 'trialID',
                whis = [0,100], palette = colourMap, width = 0.75,
                zorder = 5, ax = ax[vv])
    
    #Alter the faces and outlines of bars
    #Loop through boxes and fix colours
    for ii in range(len(ax[vv].artists)):
    
        #Get the current artist
        artist = ax[vv].artists[ii]
        
        #Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        artist.set_edgecolor(col)
        artist.set_facecolor('None')
        
        #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        #Loop over them here, and use the same colour as above
        #The first two relate to the whisker lines, so we set these to dashes
        for jj in range(ii*6,ii*6+6):
            line = ax[vv].lines[jj]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            if jj < ii*6 + 2:
                line.set_linestyle('--')
                
    #Add violin plot
    sns.violinplot(data = df_currData,
                   x = 'extractNo', y = 'peakAbsError', hue = 'trialID',
                   cut = True, scale = 'width', inner = None,
                   palette = colourMap, width = 0.75,
                   zorder = 4, ax = ax[vv])
    
    #Adjust alpha and edge width on violins
    for violin in ax[vv].collections:
        violin.set_alpha(0.3)
        violin.set_linewidth(0)
        
    #Add point plot with mean and standard deviation
    sns.pointplot(data = df_currData,
                  x = 'extractNo', y = 'peakAbsError', hue = 'trialID', ci = 'sd',
                  join = False, palette = colourMap2,
                  scale = 0.6, errwidth = 1.5, dodge = 0.5,
                  zorder = 5, ax = ax[vv])

    #Adjust x-axes labels
    if vv == len(analysisVar)-1:
        #Set x-limits
        ax[vv].set_xlim([ax[vv].get_xlim()[0]-0.5, ax[vv].get_xlim()[1]+0.5])
        #Set x-ticks
        ax[vv].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/1)+1))
        #Set label
        ax[vv].set_xlabel('No. of Gait Cycles')
    else:
        #Set x-limits
        ax[vv].set_xlim([ax[vv].get_xlim()[0]-0.5, ax[vv].get_xlim()[1]+0.5])
        #Set x-ticks
        ax[vv].set_xticks(np.linspace(0,len(extractNoDual)-1, int((len(extractNoDual)-1)/1)+1))
        #Remove x-ticks
        ax[vv].xaxis.set_ticklabels([])
        #Remove label
        ax[vv].set_xlabel('')
        
    #Turn off legend
    ax[vv].get_legend().remove()
        
    #Set y-label
    ax[vv].set_ylabel('Peak Abs. Difference (\u00b0)')
    
    #Set title
    ax[vv].set_title(f'1D {analysisLabels[vv]}')

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_ALL_1D.pdf')
fig.savefig(f'{samplesCompDir}\\Figures\\AbsoluteError_NoGaitCycle_ALL_1D.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()
    
# %% ----- End of analyseFukuchi2017-Running_RQ3.py -----
    