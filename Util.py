import pandas as pd
import numpy as np
from sklearn import cross_validation
from Constants import *

#Normalize def taken from hmm.py library in the sklearn toolkit.
EPS = np.finfo(float).eps
def normalize(A, axis=None):
    """ Normalize the input array so that it sums to 1.

    Parameters
    ----------
    A: array, shape (n_samples, n_features)
       Non-normalized input data
    axis: int
          dimension along which normalization is performed

    Returns
    -------
    normalized_A: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies inplace the array
    """
    A += EPS
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum
    
def nFoldCrossValidator(model,train,target):
    """Cross validate the model using 20-fold cross validation. This method uses the KFold cross validation from sklearn toolkit
    
    Parameters
    ----------
    model: The trained model used for classification
    train: array shape[n_samples,n_features] 
        
    target: array with shape [n_samples] 
    
    Returns
    --------
    score: Cross validated score as a percentage. (float)"""
    #Setting up cross validation
    kf_total = cross_validation.KFold(len(train), n_folds=20, shuffle=False, random_state=6)
    scoreArray = cross_validation.cross_val_score(model, train, target, cv=kf_total, n_jobs = 1)
    
    probabilitiesList = []
    predictedList = []
    newTarget = []

#==============================================================================
    #Find the positions for the missing states in dataset
    activities = np.unique(target)
    missingPos = [] 
    if len(activities) < 6:
        if 'Lying' not in activities:
            missingPos.append(0)
        if 'Sitting' not in activities:
            missingPos.append(1)
        if 'StairClimbing' not in activities:
            missingPos.append(2)
        if 'Standing' not in activities:
            missingPos.append(3)
        if 'Walking' not in activities:
            missingPos.append(4)
        if 'Wheeling' not in activities:
            missingPos.append(5)
#==============================================================================
    for train_index, test_index in kf_total:
        model.fit(train[train_index], target[train_index])
        probas = model.predict_proba(train[test_index])
#==============================================================================
      #Fixing the missing states in dataset by filling with 0.0001       
        if len(missingPos) > 0:
            value = 0.0001
            for x in range(len(probas)):
                    probas[x][np.argmin(probas[x])] = probas[x][np.argmin(probas[x])] - (value * len(missingPos))
            for i in missingPos:
                
                probas = np.insert(probas, i, value, 1)
#==============================================================================
        probabilitiesList.append(probas)
        predicted = model.predict(train[test_index])
        predictedList.append(predicted)
        newTarget.append(target[test_index])

    score = np.mean(scoreArray)*100
    
    predictedList = [item for sublist in predictedList for item in sublist]
    probabilitiesList = [item for sublist in probabilitiesList for item in sublist]
    newTarget = [item for sublist in newTarget for item in sublist]
    return predictedList, probabilitiesList, score, newTarget
    
def getFramesList():
    """
    A method to retrieve all the data frames of all subjects and convert the DateTime object from string to object DateTime.
    
    Returns
    --------
    dataFrames
    """
    dataFrames = []
    path = Constants.PreprocessedDataFrames+'DataFrame_Subject_'
    for i in range(1,14):
        df = pd.read_csv(path+str(i)+'.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        dataFrames.append(df)
    return dataFrames
    
def getFTArrays():
    """
    A method that reads the features and target activities from the subjects file and returns it.
    
    Returns
    ---------
    targetMaster: A list of lists where each sub-list contains the target activities corresponding to the features. Size of list = Number of subjects
    featuresMaster: A list of lists where each sub-list contains the features corresponding to the target activities. Size of the main list = Number of subjects
    """
    targetMaster = []
    featuresMaster = []
    path = Constants.FEATURES_TARGET + 'Subject_'
    for i in range(1,14):
        df = pd.read_csv(path+str(i)+'.csv', error_bad_lines=False)
        target = np.array(df['target'].tolist())
        targetMaster.append(target)
        features = np.array(df[['meanX','abs(meanX)','meanY','abs(meanY)','meanZ','abs(meanZ)','skewX','skewY','skewZ','kurtosisX','kurtosisY','kurtosisZ','stdX','stdY','stdZ','rmsX','rmsY','rmsZ','minX','minY','minZ','maxX','maxY','maxZ','absMinX','absMinY','absMinZ','absMaxX','absMaxY','absMaxZ','meanXY','meanXZ','meanYZ','absMeanXY','absMeanXZ','absMeanYZ','overallMeanAcceleration']])
        featuresMaster.append(features)

    #After reading everything, we need to clean the data. Some subjects may have classes that are not in other subjects. 
    #This would induce an inevitable error
    featuresMaster, targetMaster = cleanData(featuresMaster, targetMaster)

    return featuresMaster,targetMaster
    

def cleanData(featuresMaster, targetMaster):
    #After reading everything, we need to clean the data. Some subjects may have classes that are not in other subjects. This would induce an
    #inevitable error
    dict_list = []
    for i in range(len(targetMaster)):
        dict_list.append({})

    for i in range(len(targetMaster)):
        for j in range(len(targetMaster[i])):
            if targetMaster[i][j] not in dict_list[i]:
                dict_list[i][targetMaster[i][j]] = 0
            dict_list[i][targetMaster[i][j]] += 1
    
    for i in range(len(dict_list)):
        for key in dict_list[i]:
            can_be_learned = False
            for j in range(len(dict_list)):
                if i == j:
                    continue
                if key in dict_list[j]:
                    can_be_learned = True
                    break
            if not can_be_learned:
                index_list = []
                for j in range(len(targetMaster[i])):
                    if key == targetMaster[i][j]:
                        index_list.append(j)
                targetMaster[i] = np.delete(targetMaster[i], index_list, 0)
                featuresMaster[i] = np.delete(featuresMaster[i], index_list, 0)

    return featuresMaster, targetMaster

def writeToExcel(dataFram,filename):
    '''Writes the data to an excel file'''
    writer = pd.ExcelWriter(filename, date_format='hh:mm:ss.000')
    dataFrame.to_excel(writer)
    writer.close()
    #print dataFrame
    
            
def writeToCSV(dataFrame,filename):
    '''Writes the data to a csv file'''
    dataFrame.to_csv(filename)
    
