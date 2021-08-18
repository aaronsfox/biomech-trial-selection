# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:29:26 2021

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    RQ4: Cycle sampling and number effect on refuting null hypothesis

    This section examines the effect of gait cycle number and sampling on the findings
    from null hypothesis testing, specifically the effect of speed on gait biomechanics.
    The original work of Fukuchi et al. (2017) examined this question, and found 
    a series of biomechanical variables are impacted by gait speed. Here we iteratively
    replicate this hypothesis testing with varying numbers and differently sampled
    gait cycles contributing to participant data. The theory being tested here is
    how does the number and selection of gait cycles impact the answers to our hypothesis
    testing.
        
    To start with, replicate the Fukuchi et al. analysis...
    
"""

# %% Import packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import _pickle as cPickle
import bz2
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import spm1d
import warnings
warnings.filterwarnings('ignore') #filter warnings for SPM1D
import time

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

#Set equal variance to true for statistical tests
equalVar = True

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

# %% Set sampling values

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

# %% Run ANOVA & post-hoc on sampled data

#### TODO: set boolean operator for whether to run stats...

#### 1D stats pickle outputs are too large - do we need the data?
#### Still big even without the data, need the SPM objects?

#Loop through variables
for vv in range(len(analysisVar)):
    
    #Loop through gait cycle extraction numbers
    for ee in range(len(extractNo)):
        
        #Start timer for current extraction number and variable
        t0 = time.clock()
        
        #Set current extraction number
        currNo = int(extractNo[ee])

        #Set generic anova dictionary to store each trial in
        aoData = []
        
        #Set dictionary to store 0D ANOVA results
        anovaDict_0D = {'extractNo': [], 'analysisVar': [],
                        #'analysisData': [], 'mean': [], 'sd': [],
                        'aovrmResults': [], 'F': [], 'p': [], 'rejectH0': [],
                        'pairwiseComp': []}
        
        #Set dictionary to store 1D ANOVA results
        anovaDict_1D = {'extractNo': [], 'analysisVar': [],
                        'Fi': [],
                        #'mean': [], 'sd': [],
                        #'z': [], 'zstar': [],
                        'rejectH0': [], 'pSet': [], 'pCluster': [], 'clusters': [],
                        'pairwiseComp': []}
        
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
            analysisDict_1D = {'subID': [], 'val': [], 'speed': []}
            
            #Set dictionary to store 0D post-hoc pairwise results
            pairwiseDict_0D = {'extractNo': [], 'analysisVar': [],
                               'comparison': [], #'mean': [], 'sd': [],
                               't': [], 'p': [], 'rejectH0': []}
            
            #Set dictionary to store 1D post-hoc pairwise results
            pairwiseDict_1D = {'extractNo': [], 'analysisVar': [],
                               'comparison': [], #'mean': [], 'sd': [],
                               'ti': [], #'z': [],
                               'pSet': [], 'p': [], 'clusters': [],
                               'rejectH0': []}
            
            #Loop through trials and append data into analysis dictionary
            for tt in range(len(trialList)):
            
                #Extract current sample for 0D data
                val0D = aoData[tt].loc[(aoData[tt]['sampleNo'] == ss) &
                                       (aoData[tt]['varType'] == '0D'),['vals']].values[0][0]
                val1D = aoData[tt].loc[(aoData[tt]['sampleNo'] == ss) &
                                       (aoData[tt]['varType'] == '1D'),['vals']].values[0][0]
                
                #Append values with appropriate identifiers to analysis dictionary
                for ii in range(len(subList)):
                    #0D
                    analysisDict_0D['subID'].append(subList[ii])
                    analysisDict_0D['val'].append(val0D[ii])
                    analysisDict_0D['speed'].append(trialList[tt])
                    #1D
                    analysisDict_1D['subID'].append(ii)
                    analysisDict_1D['val'].append(val1D[ii,:])
                    analysisDict_1D['speed'].append(tt)
            
            #0D analyses    
            
            #Convert analysis dictionary to dataframe
            df_analysis_0D = pd.DataFrame.from_dict(analysisDict_0D)
            df_analysis_1D = pd.DataFrame.from_dict(analysisDict_1D)
                    
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
            # anovaDict_0D['analysisData'].append(df_analysis_0D)
            # anovaDict_0D['mean'].append(df_analysis_0D.groupby('speed').mean()['val'])
            # anovaDict_0D['sd'].append(df_analysis_0D.groupby('speed').std()['val'])
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
                    # pairwiseDict_0D['mean'].append([np.mean(y1),np.mean(y2)])
                    # pairwiseDict_0D['sd'].append([np.std(y1),np.std(y2)])
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
            
            #1D analyses
            
            #Place data in array
            Y = np.stack(analysisDict_1D['val'], axis = 0)
            
            #Extract grouping and subject variables
            A = np.array(analysisDict_1D['speed'])
            SUBJ = np.array(analysisDict_1D['subID'])
            
            # #Calculate mean and SD for groups
            # #Place in a pandas series
            # mean1D = pd.Series(data = {trialList[0]: np.mean(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 0,]['val'].to_numpy(), axis = 0), axis = 0),
            #                            trialList[1]: np.mean(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 1,]['val'].to_numpy(), axis = 0), axis = 0),
            #                            trialList[2]: np.mean(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 2,]['val'].to_numpy(), axis = 0), axis = 0)},
            #                    index = trialList, name = 'val')
            # sd1D = pd.Series(data = {trialList[0]: np.std(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 0,]['val'].to_numpy(), axis = 0), axis = 0),
            #                          trialList[1]: np.std(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 1,]['val'].to_numpy(), axis = 0), axis = 0),
            #                          trialList[2]: np.std(np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == 2,]['val'].to_numpy(), axis = 0), axis = 0)},
            #                  index = trialList, name = 'val')
            # mean1D.index.name = 'speed'
            # sd1D.index.name = 'speed'
            
            #Run ANOVA
            F = spm1d.stats.anova1rm(Y, A, SUBJ, equalVar)
            Fi = F.inference(alpha)
            
            #Store ANOVA results in dictionary
            anovaDict_1D['extractNo'].append(currNo)
            anovaDict_1D['analysisVar'].append(analysisVar[vv])
            # anovaDict_1D['mean'].append(mean1D)
            # anovaDict_1D['sd'].append(sd1D)
            anovaDict_1D['Fi'].append(Fi)
            # anovaDict_1D['z'].append(Fi.z)
            # anovaDict_1D['zstar'].append(Fi.zstar)
            anovaDict_1D['rejectH0'].append(Fi.h0reject)
            anovaDict_1D['pSet'].append(Fi.p_set)
            anovaDict_1D['pCluster'].append(Fi.p)  
            anovaDict_1D['clusters'].append(Fi.clusters)
            
            #Get and run post-hoc if appropriate            
            #Loop through pairwise trial comparisons
            for pp in range(len(trialList)-1):
                for qq in range(pp+1,len(trialList)):
                    
                    #Extract arrays to compare
                    yA = np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == pp,'val'].to_numpy(), axis = 0)
                    yB = np.stack(df_analysis_1D.loc[df_analysis_1D['speed'] == qq,'val'].to_numpy(), axis = 0)
                    
                    if Fi.h0reject:
                    
                        #Compare
                        t = spm1d.stats.ttest_paired(yA, yB)
                        ti = t.inference(alpha, two_tailed = True, interp = True)
                        
                    #Append results to dictionary
                    
                    #1D
                    pairwiseDict_1D['extractNo'].append(currNo)
                    pairwiseDict_1D['analysisVar'].append(analysisVar[vv])
                    pairwiseDict_1D['comparison'].append([trialList[pp],trialList[qq]])
                    # pairwiseDict_1D['mean'].append([np.mean(yA, axis = 0), np.mean(yB, axis = 0)])
                    # pairwiseDict_1D['sd'].append([np.std(yA, axis = 0), np.std(yB, axis = 0)])
                    if Fi.h0reject:
                        pairwiseDict_1D['ti'].append(ti)   
                        # pairwiseDict_1D['z'].append(ti.z)
                        pairwiseDict_1D['pSet'].append(ti.p_set)
                        pairwiseDict_1D['p'].append(ti.p)
                        pairwiseDict_1D['clusters'].append(ti.clusters)
                        if ti.h0reject:
                            pairwiseDict_1D['rejectH0'].append(True)
                        else:
                            pairwiseDict_0D['rejectH0'].append(False)           
                    else:
                        pairwiseDict_1D['ti'].append(np.nan)   
                        pairwiseDict_1D['z'].append(np.nan)
                        pairwiseDict_1D['pSet'].append(np.nan)
                        pairwiseDict_1D['p'].append(np.nan)
                        pairwiseDict_1D['clusters'].append(np.nan)
                        pairwiseDict_1D['rejectH0'].append(np.nan)
                        
            #Append pairwise comparisons to ANOVA dictionary
            anovaDict_1D['pairwiseComp'].append(pairwiseDict_1D)
                        
            # #Print confirmation
            # print('Completed speed comparison '+str(ss+1)+' of '+str(nSamples)+' for '+
            #       str(currNo)+' gait cycles of '+analysisVar[vv])
            
        #Save anova and post-hoc comparison dictionary for current iteration
        #0D
        savePickleDict(anovaDict_0D, diffCompDir+'\\Fukuchi2017-Running-stats0D-n'+ \
                       str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'.pbz2')
        #1D
        savePickleDict(anovaDict_1D, diffCompDir+'\\Fukuchi2017-Running-stats1D-n'+ \
                       str(nSamples)+'-gc'+str(currNo)+'-'+analysisVar[vv]+'.pbz2')
            
        #End timer for current extraction number and variable
        t1 = time.clock()
        
        #Calculate total time this section took
        ##### TODO: this is not working!!!
        iterTime = t1 - t0
        
        #### TODO: create estimate of how long is remaining
        #### Will need an overall time variable


#### Does the null hypothesis rejection rate need to be contrasted with a power
#### analysis of sorts to discover our true potential discovery rate?????

####Could do Kappa agreement between sample conclusions?????

