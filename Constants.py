import os

class Constants():
    path = os.path.dirname(os.path.abspath(__file__))
    
    #----------------Used in GetFeaturesTarget.py-------------------#
    #FeaturesTarget folder path
    FEATURES_TARGET = path+'/FeaturesTarget/'
    
    
    #-----Used in Results.py and HMMCode.py----------------#
    SubjectWise = 'SubjectWise'
    TenFold = 'TenFold'
    WithinSubjectwise = 'WithinSubjectwise'
    ResultsFolderList = [SubjectWise,TenFold]
    HMMFolder = 'HMM'
    StaticFolder = 'Static'
    
    FINAL_RESULTS_FOLDER = path + '/Results/'
    
        
    #----------PreprocessData.py-------------------#
    PreprocessedDataFrames = path+'/DataFrames/'
    
    
    #--------------SubjectFiles---------------------------#
    SubjectFilesPath = path+'/PatientData/Pre'
