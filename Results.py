import pandas as pd
import numpy as np
from Constants import *
from os import path

#global count
TC =  0

def getDataFrame(filePath):
    global TC
    '''Get the data frame from csv file specified by the filePath'''    
    if path.exists(filePath) and path.isfile(filePath):
        dataFrame = pd.read_csv(filePath)
        a = (len(dataFrame['target']))
        TC += a
        return dataFrame
        
    else:
        return None

def getAggregatedDataFrame(subjectFiles,folder):
    df = pd.DataFrame()
    for myFile in subjectFiles:
        df = df.append(getDataFrame(myFile))
        #df = df.reset_index()
    filename = Constants.FINAL_RESULTS_FOLDER+folder+'.csv'
    df.to_csv(filename)
    return df

def getFiles(folder):
    filesList = []
    fullPath = Constants.FINAL_RESULTS_FOLDER+folder+'\Subject_'
    for i in range(1,14):
        filePath = fullPath+str(i)+'.csv'
        filesList.append(filePath)
    return filesList

#Just a function to compare the lists.
# @Author: Ranulfo Neto
def comp(A, B):
    mySum = 0
    for x in range(len(B)):
        if A[x] == B[x]:
            mySum += 1
    return (mySum/float(len(B)))
    
def submain(folder):
    files = getFiles(folder)
    dataFrame = getAggregatedDataFrame(files,folder)
    activities = np.unique(dataFrame['target'])
    StaticDataFrame = pd.DataFrame(index=activities, columns=activities).fillna(0)
    HMMDataFrame = pd.DataFrame(index=activities, columns=activities).fillna(0)
        
    totalRows = len(dataFrame['target'])
    StaticStates = (dataFrame['StaticClassifierStates'].tolist())
    HmmStates = (dataFrame['HmmStates'].tolist())
    TargetStates = (dataFrame['target'].tolist())
        
    for i in range(totalRows):
        static_classifier_state = StaticStates[i]
        hmm_state = HmmStates[i]
        target_state = TargetStates[i]  
        StaticDataFrame[static_classifier_state][target_state] += 1 
        HMMDataFrame[hmm_state][target_state] += 1 
    
    
    print(("-------Validation Type: "+folder+"---------------"))
    
    print("Static Classifier Results")
    print(StaticDataFrame)
    print()
    print(("Static Classifier Accuracy (Non-cross validated)"+str(comp(StaticStates,TargetStates)*100)+"%"))
    print()
    print()
    print("Hybrid Approach Results")
    print(HMMDataFrame)
    print()
    print(("HMM Accuracy "+str(comp(HmmStates,TargetStates)*100)+"%"))
    print()
    print()
    
    
def main():
    folderList = Constants.ResultsFolderList
    for folder in folderList:
        submain(folder)    

main()
    