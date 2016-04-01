import sys
sys.path.insert(0, '../')

import unittest
import pandas as pd
import numpy as np
import Util

class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    #Testing if the function getFTArrays() is returning the correct structures
    #One test is applied. This test checks if the types of the returned 
    #structure match the expected ones
    def test_getFTArrays(self):
        features, classes = Util.getFTArrays()
        
        self.assertTrue(type(features) is list and 
                        type(features[0]) is np.ndarray and
                        type(features[0][0]) is np.ndarray and
                        type(features[0][0][0]) is np.float64 and
                        type(classes) is list and
                        type(classes[0]) is np.ndarray and
                        type(classes[0][0]) is np.string_)
    
    def test_bug1fix(self):
        targetMaster = []
        targetMaster.append(np.array(["A","B","C","D"]))
        targetMaster.append(np.array(["A","B"]))
        targetMaster.append(np.array(["B","C"]))
        targetMaster.append(np.array(["A","K"]))

        featuresMaster = []
        featuresMaster.append(np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]))
        featuresMaster.append(np.array([[1,2,3], [4,5,6]]))
        featuresMaster.append(np.array([[4,5,6], [7,8,9]]))
        featuresMaster.append(np.array([[1,2,3], [13,14,15]]))

        #After reading everything, we need to clean the data. Some subjects may have classes that are not in other subjects. 
        #This would induce an inevitable error
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

        #Checking correctness
        dict_list = []
        for i in range(len(targetMaster)):
            dict_list.append({})

        for i in range(len(targetMaster)):
            for j in range(len(targetMaster[i])):
                if targetMaster[i][j] not in dict_list[i]:
                    dict_list[i][targetMaster[i][j]] = 0
                dict_list[i][targetMaster[i][j]] += 1
        
        correct = True
        can_be_learned = False
        for i in range(len(dict_list)):
            for key in dict_list[i]:
                for j in range(len(dict_list)):
                    if i == j:
                        continue
                    if key in dict_list[j]:
                        can_be_learned = True
                        break
                if not can_be_learned:
                    correct = False
        
        self.assertTrue(correct)

        same_size = True
        if len(targetMaster) != len(featuresMaster):
            same_size = False

        self.assertTrue(same_size)

        for i in range(len(targetMaster)):
            if len(targetMaster[i]) != len(featuresMaster[i]):
                same_size = False
                break

        self.assertTrue(same_size)

if __name__ == '__main__':
    unittest.main()
