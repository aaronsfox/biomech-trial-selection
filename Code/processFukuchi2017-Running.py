# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:06:05 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function works through the Fukuchi et al. (2017) running C3D data and
    processes the marker/GRF data using Opensim to extract joint kinematics and
    kinetics using inverse kinematics and dynamics.
    
    Specific notes on analysis processes are outlined in the comments.
    
"""

# %% Import packages

import opensim as osim
import os
import pandas as pd
import glob
import shutil
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# %% Define functions


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
            
#Set generic opensim parameters
#Note that scaling paths are relative to each participants directory

#Set generic model path
genModelPath = '..\\..\\OpensimModel'

#Set generic model file
genModelFileName = genModelPath+'\\LaiArnold2017_refined_Fukuchi2017-running.osim'

#Add model geometry to search paths
osim.ModelVisualizer.addDirToGeometrySearchPaths(genModelPath+'\\Geometry')

#Create the measurement set object for later use
measurementSetObject = osim.OpenSimObject.makeObjectFromFile('..\\OpensimModel\\scaleMeasurementSet_Fukuchi2017-running.xml')
measurementSet = osim.MeasurementSet.safeDownCast(measurementSetObject)

#Set the scale task set
scaleTaskSet = osim.IKTaskSet('..\\OpensimModel\\scaleTasks_Fukuchi2017-running.xml')

#Set scale order
scaleOrder = osim.ArrayStr()
scaleOrder.set(0,'measurements')

#Set IK task set
ikTaskSet = osim.IKTaskSet('..\\OpensimModel\\ikTasks_Fukuchi2017-running.xml')

# %% Loop through subjects and process data

#Set-up lists to store opensim objects in
#This seems necessary to doubling up on objects and crashing Python
scaleTool = []
# osimModel = []
ikTool = []
# analyseTool = []
# bodyKin = []
# anToolRun = []

#Loop through subjects
#Set starting subject index to account for starting at an index > 0 in lists
startInd = 0
for ii in range(len(subList)):
    
    #Navigate to subject directory
    os.chdir(subList[ii])
    
    # %% Model scaling
    
    #Identify static trial name
    staticFile = []
    for file in glob.glob('*static.c3d'):
        staticFile.append(file)
    
    #Get subject mass based on static file name and participant info database
    mass = df_subInfo.loc[df_subInfo['FileName'] == staticFile[0],['Mass']].values[0][0]
    
    #Convert c3d markers to trc file
    
    #Set-up data adapters
    c3d = osim.C3DFileAdapter()
    trc = osim.TRCFileAdapter()
    
    #Get markers
    static = c3d.read(staticFile[0])
    markersStatic = c3d.getMarkersTable(static)
    
    #Write static data to file
    trc.write(markersStatic, staticFile[0].split('.')[0]+'.trc')
    
    #Set variable for static trial trc
    staticTrial_trc = staticFile[0].split('.')[0]+'.trc'
    
    #Set scale time range
    scaleTimeRange = osim.ArrayDouble()
    scaleTimeRange.set(0,osim.Storage(staticTrial_trc).getFirstTime())
    scaleTimeRange.set(1,osim.Storage(staticTrial_trc).getFirstTime()+0.5)
    
    #Set-up scale tool    
    scaleTool.append(osim.ScaleTool())
    
    #Set mass in scale tool
    scaleTool[ii-startInd].setSubjectMass(mass)
    
    #Set generic model file
    scaleTool[ii-startInd].getGenericModelMaker().setModelFileName(genModelFileName)
    
    #Set the measurement set
    scaleTool[ii-startInd].getModelScaler().setMeasurementSet(measurementSet)
    
    #Set scale tasks
    for k in range(0,scaleTaskSet.getSize()-1):
        scaleTool[ii-startInd].getMarkerPlacer().getIKTaskSet().adoptAndAppend(scaleTaskSet.get(k))
    
    #Set marker file
    scaleTool[ii-startInd].getMarkerPlacer().setMarkerFileName(staticTrial_trc)
    scaleTool[ii-startInd].getModelScaler().setMarkerFileName(staticTrial_trc)
    
    #Set options
    scaleTool[ii-startInd].getModelScaler().setPreserveMassDist(True)
    scaleTool[ii-startInd].getModelScaler().setScalingOrder(scaleOrder)
    
    #Set time ranges
    scaleTool[ii-startInd].getMarkerPlacer().setTimeRange(scaleTimeRange)
    scaleTool[ii-startInd].getModelScaler().setTimeRange(scaleTimeRange)
    
    #Set output files
    scaleTool[ii-startInd].getModelScaler().setOutputModelFileName(subList[ii]+'_scaledModel.osim')
    scaleTool[ii-startInd].getModelScaler().setOutputScaleFileName(subList[ii]+'_scaleSet.xml')
    
    #Set marker adjuster parameters
    scaleTool[ii-startInd].getMarkerPlacer().setOutputMotionFileName(staticTrial_trc.split('.')[0]+'.mot')
    scaleTool[ii-startInd].getMarkerPlacer().setOutputModelFileName(subList[ii]+'_scaledModelAdjusted.osim')
    
    #Print and run scale tool
    scaleTool[ii-startInd].printToXML(subList[ii]+'_scaleSetup.xml')
    scaleTool[ii-startInd].run()
    
    #Print confirmation
    print('Scaling complete for '+subList[ii])
    
    # %% Inverse kinematics
    
    #Identify static trial name
    dynamicFiles = []
    for file in glob.glob('*run*.c3d'):
        dynamicFiles.append(file)
        
    #Create subject storage lists within tools current subject
    ikTool.append([])
    
    #Loop through trials
    for tt in range(len(dynamicFiles)):
        
        #Convert c3d to TRC file
        
        #Set-up data adapters
        c3d = osim.C3DFileAdapter()
        trc = osim.TRCFileAdapter()
        
        #Get markers
        dynamic = c3d.read(dynamicFiles[tt])
        markersDynamic = c3d.getMarkersTable(dynamic)
        
        #Write data to file (unfiltered)
        trc.write(markersDynamic, dynamicFiles[tt].split('.')[0]+'.trc')
        
        #Filter marker data at 10 Hz like in paper
                
        #Define filter (10Hz low-pass 4th Order Butterworth)
        fs = float(markersDynamic.getTableMetaDataAsString('DataRate'))
        fc = 10.0
        w = fc / (fs / 2.0)
        order = 2 #4th order with bi-directional filter
        b,a = butter(order, w, btype = 'low', analog = False)
        
        #Flatten table to filter
        markersFlat = markersDynamic.flatten()
        
        #Get time in array format
        t = np.array(markersFlat.getIndependentColumn())
        
        #Create numpy array to store filtered data in
        filtData = np.empty((markersFlat.getNumRows(),
                             markersFlat.getNumColumns()))
        
        #Loop through column labels
        for cc in range(len(markersFlat.getColumnLabels())):
            
            #Get the data in an array format
            datArr = markersFlat.getDependentColumn(markersFlat.getColumnLabels()[cc]).to_numpy()
            
            #Interpolate across nans if present
            if np.sum(np.isnan(datArr)) > 0:
            
                #Remove nans from data array and corresponding times
                yi = datArr[~np.isnan(datArr)]
                xi = t[~np.isnan(datArr)]
                
                #Interpolate data to fill gaps
                cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
                dVals = cubicF(np.linspace(t[0], t[-1], len(t)))
                
            else:
                
                #Set data values to the original array
                dVals = datArr
            
            #Filter data
            dFilt = filtfilt(b, a, dVals)
            
            #Set filtered data back in data array
            filtData[:,cc] = dFilt
            
        #Create blank time series table to fill
        filtMarkers = osim.TimeSeriesTable()
        
        #Fill row vector with data and append to the timeseries table
        #Loop through rows
        for rr in range(len(t)):
            #Create row vector from current row of array
            row = osim.RowVector.createFromMat(filtData[rr,:])
            #Append row to table
            filtMarkers.appendRow(rr,row)
            #Set time value
            filtMarkers.setIndependentValueAtIndex(rr,t[rr])
        
        #Set column labels
        filtMarkers.setColumnLabels(markersDynamic.flatten().getColumnLabels())
        
        #Pack table back to Vec3
        filtTRC = filtMarkers.packVec3()
        
        #Write meta data
        filtTRC.addTableMetaDataString('DataRate',str(fs)) #Data rate
        filtTRC.addTableMetaDataString('CameraRate',str(fs)) #Camera rate
        filtTRC.addTableMetaDataString('NumFrames',str(len(t))) #Number frames
        filtTRC.addTableMetaDataString('NumMarkers',str(len(markersDynamic.getColumnLabels()))) #Number markers
        filtTRC.addTableMetaDataString('Units',markersDynamic.getTableMetaDataString('Units')) #Units
        
        #Write data to file (unfiltered)
        trc.write(filtTRC, dynamicFiles[tt].split('.')[0]+'_filtered.trc')
        
        #Set variable for dynamic trial trc
        dynamicTrial_trc = dynamicFiles[tt].split('.')[0]+'_filtered.trc'
        
        #Initialise IK tool
        ikTool[ii-startInd].append(osim.InverseKinematicsTool())
        
        #Set the model
        ikTool[ii-startInd][tt].set_model_file(subList[ii]+'_scaledModelAdjusted.osim')
        
        #Set task set
        ikTool[ii-startInd][tt].set_IKTaskSet(ikTaskSet)
        
        #Set marker file
        ikTool[ii-startInd][tt].set_marker_file(dynamicTrial_trc)
        
        #Set times
        ikTool[ii-startInd][tt].setStartTime(osim.Storage(dynamicTrial_trc).getFirstTime())
        ikTool[ii-startInd][tt].setEndTime(osim.Storage(dynamicTrial_trc).getLastTime())
        
        # #Set error reporting to false
        # ikTool[ii-startInd][tt].set_report_errors(False)
        
        #Set output filename
        ikTool[ii-startInd][tt].set_output_motion_file(dynamicFiles[tt].split('.')[0]+'_ik.mot')
        
        #Print and run tool
        ikTool[ii-startInd][tt].printToXML(dynamicFiles[tt].split('.')[0]+'_setupIK.xml')
        ikTool[ii-startInd][tt].run()
        
        #Rename marker errors file
        shutil.move('_ik_marker_errors.sto',dynamicFiles[tt].split('.')[0]+'_ik_marker_errors.sto')
        
        #Print confirmation
        print('IK complete for '+dynamicFiles[tt].split('.')[0])
    
    #Print confirmation for subject
    print('IK complete for '+subList[ii])
    
    # %%
    
    #Shift back up to data directory
    os.chdir(dataDir)
    
