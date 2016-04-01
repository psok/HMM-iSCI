import os

class Constants():
    path = os.path.dirname(os.path.abspath(__file__))
    
    #----------------Used in GetFeaturesTarget.py-------------------#
    #FeaturesTarget folder path
    FEATURES_TARGET = path+'/FeaturesTarget/'
    
    
    #-----Used in Results.py and HMMCode.py----------------#
    SubjectWise = 'SubjectWise'
    TenFold = 'TenFold'
    ResultsFolderList = [SubjectWise,TenFold]
    
    FINAL_RESULTS_FOLDER = path+'/FinalResults/'
    
        
    #----------PreprocessData.py-------------------#
    PreprocessedDataFrames = path+'/DataFrames/'
    
    
    #--------------SubjectFiles---------------------------#
    SubjectFilesPath = path+'/PatientData/Pre'
