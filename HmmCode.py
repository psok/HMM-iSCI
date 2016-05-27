# Created Asma Mehjabeen <amehjabeen@luc.edu>
# Modified by Pichleap Sok (Jessie) <psok@luc.edu>
import pandas as pd
from HMMClassification import *  
from StaticClassification import * 
import datetime
import time

def saveToCsv(dataframe, filename):
    dataframe.to_csv(filename)

def errorMatrix(statesList, targetMaster):
    #Getting all states
    ordered_states = [item for sublist in statesList for item in sublist]    
    ordered_states = np.array(ordered_states)
    ordered_states = np.unique(ordered_states)
    ordered_states = ordered_states.tolist()

    no_states = len(ordered_states)
    ordered_states_dict = {}
    for i in range(no_states):
        ordered_states_dict[ordered_states[i]] = i

    states_matrix = []
    for i in range(no_states):
        states_matrix.append([0 for j in range(no_states)])

    for i in range(len(statesList)):
        states = statesList[i]
        for j in range(len(states)):
            #Horizontal and vertical positions
            hpos = ordered_states_dict[states[j]]
            vpos = ordered_states_dict[targetMaster[i][j]]
            states_matrix[vpos][hpos] += 1

    # Calculate the precision for each activity: True Positive / (True Positive + False Positive)
    precision_matrix = []
    for i in range(no_states):
        precision_matrix.append([0 for j in range(no_states)])
        
    sum_row = np.sum(states_matrix, axis=1)
    for row in range(no_states):
        for col in range(no_states):
            precision_matrix[row][col] = (states_matrix[row][col] / sum_row[row])
        
    # Calculate the recall for each activity: True Positive / (True Positive + False Negative)
    recall_matrix = []
    for i in range(no_states):
        recall_matrix.append([0 for j in range(no_states)])
    sum_col = np.sum(states_matrix, axis=0)
    for col in range(no_states):
        for row in range(no_states):
            recall_matrix[row][col] = (states_matrix[row][col] / sum_col[col])
        
    # save confusion matrix, precision and recall to file respectively
    df_confusion = pd.DataFrame(states_matrix, index = ordered_states,
                          columns = ordered_states)
    df_precision = pd.DataFrame(precision_matrix, index = ordered_states,
                          columns = ordered_states)
    df_recall = pd.DataFrame(recall_matrix, index = ordered_states,
                          columns = ordered_states)  
                          
    return df_confusion, df_precision, df_recall
    
def main():    
    start_time = time.time()
    
    classifierObject = MyStaticClassifiers()
    methodNames = [classifierObject.SVMClassifier, classifierObject.logisticRegression, classifierObject.decisionTreeClassification, 
                 classifierObject.kNearestNeighborsClassifier, classifierObject.naiveBayesClassifier, classifierObject.randomForestClassifier]
#
    for methodName in methodNames:
        
        folder = methodName.__name__
        print(folder)
        validations = [Constants.TenFold, Constants.SubjectWise, Constants.WithinSubjectwise]
        for validation in validations:
            
            filePath = Constants.FINAL_RESULTS_FOLDER + folder + '/' + validation + '/'
            
            staticFilePath = filePath + Constants.StaticFolder + '/'
            if not os.path.exists(staticFilePath):       
                os.makedirs(staticFilePath) 
            
            hmm = HMMClassification()
            featuresMaster, targetMaster = getFTArrays()
            
            if validation == Constants.TenFold:
                predictedList, probabilitiesList, targetMaster = classifierObject.nFoldValidationOnSubjects(
                                                                                methodName, 
                                                                                featuresMaster, 
                                                                                targetMaster, 
                                                                                staticFilePath)
            elif validation == Constants.SubjectWise:
                predictedList, probabilitiesList = classifierObject.subjectWiseCrossValidation(
                                                                                methodName,
                                                                                featuresMaster, 
                                                                                targetMaster, 
                                                                                staticFilePath)
            else:
                predictedList, probabilitiesList, targetMaster = classifierObject.withinSubjectWiseValidation(
                                                                                methodName,
                                                                                featuresMaster, 
                                                                                targetMaster, 
                                                                                staticFilePath)                                           
        
            df_confusion, df_precision, df_recall = errorMatrix(predictedList, targetMaster)
            saveToCsv(df_confusion, filePath + "Error_Matrix.csv")
            saveToCsv(df_precision, filePath + "Precision_Matrix.csv")
            saveToCsv(df_recall, filePath + "Recall_Matrix.csv")
            print(df_confusion)
            print(df_precision)
            print(df_recall)
            
            hmmFilePath = filePath + Constants.HMMFolder + '/'
            if not os.path.exists(hmmFilePath):    
                os.makedirs(hmmFilePath)
            
            if validation == Constants.TenFold:
                HmmStatesList, targetMaster = hmm.hmmNFoldValidation(targetMaster,probabilitiesList, hmmFilePath)
            elif validation == Constants.SubjectWise:
                HmmStatesList = hmm.hmmClassifierSubjectWiseValidation(targetMaster,probabilitiesList, hmmFilePath)
            else:
                HmmStatesList, targetMaster = hmm.hmmWithinSubjectWiseValidation(targetMaster,probabilitiesList, hmmFilePath)
            
            
            df_confusion, df_precision, df_recall = errorMatrix(HmmStatesList, targetMaster)
            saveToCsv(df_confusion, filePath + "HMM_Error_Matrix.csv")
            saveToCsv(df_precision, filePath + "HMM_Precision_Matrix.csv")
            saveToCsv(df_recall, filePath + "HMM_Recall_Matrix.csv")
            print(df_confusion)
            print(df_precision)
            print(df_recall)
            
            overalResultFilename = hmmFilePath + 'Overall_Result.csv'      
                
            for i in range(len(targetMaster)):
                filename = hmmFilePath+ 'Subject_'+str(i+1)+'.csv'      
                df = pd.DataFrame(predictedList[i])
                df.columns = ['StaticClassifierStates']
                df['HmmStates'] = HmmStatesList[i]
                df['target'] = targetMaster[i]
                df.to_csv(filename)
                if i>1:
                    df.to_csv(overalResultFilename, index_col=False, header=False, mode = 'a')
                else: 
                    df.to_csv(overalResultFilename)
            elapsed_time = time.time() - start_time
            elapsed_time = datetime.timedelta(seconds=elapsed_time)
            print("\nTotal running time: " + str(elapsed_time))
            
            filename = filePath + "Duration.txt"
            with open(filename, "w") as textfile:
                textfile.write('Total running time: ' + str(elapsed_time))
main()
    
