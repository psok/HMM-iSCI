import pandas as pd
from os import path
from datetime import *
from Constants import *

def getFileList():
    print("Inside fileList")
    '''get the list of all files.'''
    fileList = [[]]
    fullPath = Constants.SubjectFilesPath
    fileList[0].append(fullPath+'\Subject_1\Subject_1_Pre_sized1_modifiedlabeled.csv')
    fileList[0].append(fullPath+'\Subject_1\Subject_1_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])    
    fileList[1].append(fullPath+'\Subject_2\Subject_2_Pre_sized1_modifiedlabeled.csv')
    fileList[1].append(fullPath+'\Subject_2\Subject_2_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[2].append(fullPath+'\Subject_3_no_stamp\Subject_3_Pre_sized1_modifiedlabeled.csv')
    fileList[2].append(fullPath+'\Subject_3_no_stamp\Subject_3_Pre_sized2_modifiedlabeled.csv')
    fileList[2].append(fullPath+'\Subject_3_no_stamp\Subject_3_Pre_sized3_modifiedlabeled.csv')
    fileList.append([])
    fileList[3].append(fullPath+'\Subject_4\Subject_4_Pre_sized1_modifiedlabeled.csv')
    fileList[3].append(fullPath+'\Subject_4\Subject_4_Pre_sized2_modifiedlabeled.csv')
    fileList[3].append(fullPath+'\Subject_4\Subject_4_Pre_sized3_modifiedlabeled.csv')
    fileList.append([])
    fileList[4].append(fullPath+'\Subject_5\Subject_5_Pre_sized1_modifiedlabeled.csv')
    fileList[4].append(fullPath+'\Subject_5\Subject_5_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[5].append(fullPath+'\Subject_6\Subject_6_Pre_sized4_modifiedlabeled.csv')
    fileList[5].append(fullPath+'\Subject_6\Subject_6_Pre_sized5_modifiedlabeled.csv')
    fileList.append([])    
    fileList[6].append(fullPath+'\Subject_7\Subject_7_Pre_sized1_modifiedlabeled.csv')
    fileList[6].append(fullPath+'\Subject_7\Subject_7_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[7].append(fullPath+'\Subject_8\Subject_8_Pre_sized1_modifiedlabeled.csv')
    fileList[7].append(fullPath+'\Subject_8\Subject_8_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[8].append(fullPath+'\Subject_9\Subject_9_Pre_sized1_modifiedlabeled.csv')
    fileList[8].append(fullPath+'\Subject_9\Subject_9_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[9].append(fullPath+'\Subject_10\Subject_10_Pre_sized1_modifiedlabeled.csv')
    fileList[9].append(fullPath+'\Subject_10\Subject_10_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[10].append(fullPath+'\Subject_11\Subject_11_Pre_sized1_modifiedlabeled.csv')
    fileList[10].append(fullPath+'\Subject_11\Subject_11_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[11].append(fullPath+'\Subject_12\Subject_12_Pre_sized1_modifiedlabeled.csv')
    fileList[11].append(fullPath+'\Subject_12\Subject_12_Pre_sized2_modifiedlabeled.csv')
    fileList.append([])
    fileList[12].append(fullPath+'\Subject_13\Subject_13_Pre_modifiedlabeled.csv')
    
    return fileList
    
    
def getFramesFromFiles(fileList):
    dataFrameList = []
    for subjectFiles in fileList:
        df = pd.DataFrame()
        for fileName in subjectFiles:
            df = df.append(getDataFrame(fileName))
        df = df.reset_index()
        dataFrameList.append(df)
    return dataFrameList
    
    
def getDataFrame(filePath):
    '''Get the data frame from csv file specified by the filePath'''    
    if path.exists(filePath) and path.isfile(filePath):
        dataFrame = pd.read_csv(filePath, header = None)
        # Specify column names in the data frame
        dataFrame.columns = ['iTime','X','Y','Z','Date','Time','Estimate','Position']
        
        #merge the date and the time columns
        dataFrame['DateTime'] = pd.to_datetime(dataFrame['Date']+dataFrame['Time'])
        dataFrame = dataFrame.drop(['Date','Time'],1)   
        #return the modified data frame
        return dataFrame
        
    else:
        return None
        
    
def writeFramesToFiles(dataFrameList):
    listLength = len(dataFrameList)
    filePath = Constants.PreprocessedDataFrames
    for i in range(listLength):
        print(("Writing data of subject "+str(i+1)))
        dataFrame = dataFrameList[i]
        dataFrame.to_csv(filePath+'DataFrame_Subject_'+str(i+1)+'.csv')
    
    
def main():
    print("main started")
    fileList = getFileList()
    dataFrameList = getFramesFromFiles(fileList)
    writeFramesToFiles(dataFrameList)
    
main()