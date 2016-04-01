import pandas as pd
from datetime import *
import numpy as np
from scipy.stats import *
from Util import *
from Constants import *

def getFeatures(dataFrame):
    #getting the total number of 2second clips in the data.
    totalRows = len(dataFrame['DateTime'])-1
    initial_time = dataFrame['DateTime'][0]
    final_time = dataFrame['DateTime'][totalRows]
    total_time = (final_time - initial_time).total_seconds()
    #num_clips represents the total number of 2 second clips
    num_clips = int((total_time/2))   #WHERE TO CHANGE CLIP SIZE
       
    row_index = 0
    #time value representing 2 seconds.
    clips_range = timedelta(seconds = 2) #WHERE TO CHANGE CLIP SIZE
    features = []
    f_index = 0
    target = []
    loops=[]
    #For each 2 second clip extract the features and target activities
    for i in range(num_clips):
        current_time = initial_time
        #dictionary for holding different activities and their count.
        activityDict = dict()
        #Lists to hold the values of X,Y and Z axes for each 2 second clip.
        X,Y,Z = [],[],[]
        
        loopCount = 0
        #Keep extracting information from rows till 2 seconds.
        while ((current_time - initial_time) < clips_range):
            #Extract activity
            estimatedActivity = dataFrame['Estimate'][row_index]
            #If activity is already in the dictionary increment its count
            if estimatedActivity in activityDict:
                activityDict[estimatedActivity] += 1
            #If not add it to the dictionary with a count of 1
            else:
                activityDict[estimatedActivity] = 1
            #Add the values of X, Y, and Z axes to the appropriate lists.
            X.append(dataFrame['X'][row_index])
            Y.append(dataFrame['Y'][row_index])
            Z.append(dataFrame['Z'][row_index])
            row_index += 1
            current_time = dataFrame['DateTime'][row_index]
            loopCount += 1
        #In the 2 second clip, if an activity occurs more than 80% of the time, then extract features and add them to the features list and add the activity to the target list.
        initial_time = dataFrame['DateTime'][row_index]
        mostRepeatedActivity = max(activityDict, key = activityDict.get)
        if (activityDict[mostRepeatedActivity]/float(loopCount)) > 0.8 and mostRepeatedActivity != 'Trash' and mostRepeatedActivity != 'Misc' :
            features.append([])
            TX,TY,TZ = np.array(X), np.array(Y), np.array(Z)
            meanX = TX.mean()
            features[f_index].append(meanX)
            features[f_index].append(np.absolute(TX).mean())
            meanY = TY.mean()
            features[f_index].append(meanY)
            features[f_index].append(np.absolute(TY).mean())
            meanZ = TZ.mean()
            features[f_index].append(meanZ)
            features[f_index].append(np.absolute(TZ).mean())
            
            features[f_index].append(skew(TX))
            features[f_index].append(skew(TY))
            features[f_index].append(skew(TZ))
            
            features[f_index].append(kurtosis(TX))
            features[f_index].append(kurtosis(TY))
            features[f_index].append(kurtosis(TZ))            
            
            features[f_index].append(TX.std())
            features[f_index].append(TY.std())
            features[f_index].append(TZ.std())
            
            features[f_index].append(np.sqrt(np.square(TX).sum()/len(TX)))
            features[f_index].append(np.sqrt(np.square(TY).sum()/len(TY)))
            features[f_index].append(np.sqrt(np.square(TZ).sum()/len(TZ)))
            
            features[f_index].append(TX.min())
            features[f_index].append(TY.min())
            features[f_index].append(TZ.min())            
            
            features[f_index].append(TX.max())
            features[f_index].append(TY.max())
            features[f_index].append(TZ.max())
            
            features[f_index].append(np.absolute(TX).min())
            features[f_index].append(np.absolute(TY).min())
            features[f_index].append(np.absolute(TZ).min())
            
            features[f_index].append(np.absolute(TX).max())
            features[f_index].append(np.absolute(TY).max())
            features[f_index].append(np.absolute(TZ).max())
            
            meanXY = (TX*TY).mean()
            meanXZ = (TX*TZ).mean()
            meanYZ = (TY*TZ).mean()
            features[f_index].append(meanXY)
            features[f_index].append(meanXZ)
            features[f_index].append(meanYZ)
            
            features[f_index].append(np.absolute(meanXY))
            features[f_index].append(np.absolute(meanXZ))
            features[f_index].append(np.absolute(meanYZ))
            
            MeansXYZ = [meanX, meanY , meanZ]
            features[f_index].append(np.mean(MeansXYZ))          
            
            
            f_index +=1
            target.append(mostRepeatedActivity)
        loops.append(loopCount)            
            
    features = np.array(features)
    target = np.array(target)
    return (features,target)  

def main():
    #Get the preprocessed data from all the subject files.
    dataFrameList = getFramesList()
    listLength = len(dataFrameList)
    filePath = Constants.FEATURES_TARGET
    #From each subject's data, extract features and write it to a different file in the FeaturesTarget folder.
    for i in range(listLength):
        print(('Value of i', i))
        dataFrame = dataFrameList[i]
        features,target = getFeatures(dataFrame)
        df = pd.DataFrame(features)
        df.columns = ['meanX','abs(meanX)','meanY','abs(meanY)','meanZ','abs(meanZ)','skewX','skewY','skewZ','kurtosisX','kurtosisY','kurtosisZ','stdX','stdY','stdZ','rmsX','rmsY','rmsZ','minX','minY','minZ','maxX','maxY','maxZ','absMinX','absMinY','absMinZ','absMaxX','absMaxY','absMaxZ','meanXY','meanXZ','meanYZ','absMeanXY','absMeanXZ','absMeanYZ','overallMeanAcceleration']
        df['target'] = target
        filename = filePath+ 'Subject_'+str(i+1)+'.csv'
        df = pd.DataFrame(df)
        df.to_csv(filename)
        
main()
