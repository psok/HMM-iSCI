import sys
sys.path.insert(0, '../')

import unittest
import pandas as pd
import numpy as np
import random
import HMMClassification
import StaticClassification
import Util

class TestHmmClassification(unittest.TestCase):

    def setUp(self):
        self.myHmm = HMMClassification.HMMClassification()

        #Creating data. This data will be used to feed the cross validation methods.
        #It is complex because it fits all possible real world data formats.
        #Data with subjects with same shapes of data and data with subjects with totally different shapes are created.
        subjects_num = 15
        entry_num = 10
        features_num = 10

        #Creating subjects that have same set of classes in their data.
        self.sameclass_subjectsfeat = []
        feat_count = 0
        for i in range(subjects_num):
            #
            subject_entries = []
            for j in range(entry_num):
                #Was chr(pattern_feat)
                pattern_feat = 97 + j
                subject_features = []
                for k in range(features_num - 2):
                    subject_features.append(feat_count)
                    feat_count -= 1

                subject_features.append(pattern_feat)
                subject_features.append(pattern_feat)
                subject_entries.append(subject_features)
            
            self.sameclass_subjectsfeat.append(subject_entries)
        
        #Creating classes
        self.sameclass_subjectsclass = []
        for i in range(subjects_num):
            subject_entries = []
            for j in range(entry_num):
                #Was chr(entry_class)
                entry_class = 65 + j
                subject_entries.append(entry_class)
            
            self.sameclass_subjectsclass.append(subject_entries)

        #Creating subjects that have different subsets of classes in their data.
        self.diffclass_subjectsfeat = []
        feat_count = 0

        entry_num_range = subjects_num
        for i in range(subjects_num):
            subject_entries = []
            entry_range = entry_num - random.randint(0, entry_num / 2)

            for j in range(entry_num_range):
                #Was chr(pattern_feat)
                pattern_feat = 97 + j
                subject_features = []
                for k in range(features_num - 2):
                    subject_features.append(feat_count)
                    feat_count -= 1

                subject_features.append(pattern_feat)
                subject_features.append(pattern_feat)
                subject_entries.append(subject_features)
            
            self.diffclass_subjectsfeat.append(subject_entries)
            if entry_num_range > 1:
                entry_num_range -= 1

        #Creating classes
        self.diffclass_subjectsclass = []
        for i in range(subjects_num):
            subject_entries = []
            for j in range(len(self.diffclass_subjectsfeat[i])):
                #Was chr(ch_class)
                ch_class = self.diffclass_subjectsfeat[i][j][-1] - 32
                #ch_class = random.randint(65,90)
                subject_entries.append(ch_class)
            self.diffclass_subjectsclass.append(subject_entries)

        #
        for i in range(len(self.sameclass_subjectsfeat)):
            self.sameclass_subjectsfeat[i] = np.array(self.sameclass_subjectsfeat[i])
        
        for i in range(len(self.sameclass_subjectsclass)):
            self.sameclass_subjectsclass[i] = np.array(self.sameclass_subjectsclass[i])

        for i in range(len(self.diffclass_subjectsfeat)):
            self.diffclass_subjectsfeat[i] = np.array(self.diffclass_subjectsfeat[i])

        for i in range(len(self.diffclass_subjectsclass)):
            self.diffclass_subjectsclass[i] = np.array(self.diffclass_subjectsclass[i])

        #Creating important structures and setting important variables
        self.classifierObject = StaticClassification.MyStaticClassifiers()
        self.methodName = self.classifierObject.decisionTreeClassification
        self.folder = 'DecisionTreeClassifier'
        
        #Cleaning data. This is how I fixed all bugs.
        self.sameclass_subjectsfeat, self.sameclass_subjectsclass = Util.cleanData(self.sameclass_subjectsfeat,
                                                                                   self.sameclass_subjectsclass)

        self.diffclass_subjectsfeat, self.diffclass_subjectsclass = Util.cleanData(self.diffclass_subjectsfeat,
                                                                                   self.diffclass_subjectsclass)

        self.statesList, self.probabilitiesList = self.classifierObject.subjectWiseCrossValidation(self.methodName, 
                                                                                                   self.diffclass_subjectsfeat,
                                                                                                   self.diffclass_subjectsclass, 
                                                                                                   self.folder)

    def tearDown(self):
        pass

    #Test wheter the class can be instantiated or not
    def test_classCreation(self):
        myTestHmm = HMMClassification.HMMClassification()
    
    #Testing the cross-validation method
    def test_subjectWiseCrossValidation(self):
        hmmStatesList = self.myHmm.hmmClassifierSubjectWiseValidation(self.diffclass_subjectsclass, self.probabilitiesList)

    #Testing wheter the adjustProbabilities method works.
    #This is obsolete. It is no longer necessary on the main code.
    def test_adjustProbabilities(self):
        self.myHmm.adjustProbabilities(self.probabilitiesList, self.statesList)
        same_size = True
        size_to_compare = len(self.probabilitiesList[0][0])
        for array in self.probabilitiesList:
            for inner_array in array:
                if len(inner_array) != size_to_compare:
                    same_size = False
                    break

        self.assertTrue(same_size)
                
                
if __name__ == '__main__':
    unittest.main()
    
