import sys
sys.path.insert(0, '../')

import unittest
import pandas as pd
import numpy as np
import HMMClassification
import StaticClassification
import HMMClassification
import random

class TestStaticClassifiers(unittest.TestCase):

  #Creating features that will be used to feed classifiers. This is training data.
  def createFeaturesArray(self):
    features = []
    for i in range(8):
      feat = [0, 0, 0]
      if i%2 != 0:
        feat[-1] = 1
      if i == 2 or i == 3 or i == 6 or i == 7:
          feat[-2] = 1
      if i == 4 or i == 5 or i == 6 or i == 7:
          feat[-3] = 1
      features.append(feat)

    return np.array(features)
  
  #Creating classes that will be used to feed classifiers. This is training data
  def createClassesArray(self):
    classes = []
    for i in range(8):
      classes.append(chr(97 + i))
    
    return np.array(classes)

  #
  def setUp(self):
      #Creating the object
      self.myStaticClassifiers = StaticClassification.MyStaticClassifiers()

      #Simple data that has the purpose of testing the classifiers
      self.features = self.createFeaturesArray()
      self.classes = self.createClassesArray()
      self.features = self.features[0:2]
      self.classes = self.classes[0:2]
      
      #Creating data that will be used to feed the cross_validation code
      #This data is not the same that was created above. The reason why I did that is because that data has the
      #purpose of being a fast way (small and obvious) to evaluate wheter the classifiers are working or not.
      #This one is more complex and serve well as a data that emulates the real world.
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
      for i in range(subjects_num):
          subject_entries = []
          new_range = entry_num - random.randint(0, entry_num/2)
          for j in range(entry_num - random.randint(0, entry_num/2)):
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

      #Creating classes
      self.diffclass_subjectsclass = []
      for i in range(subjects_num):
          subject_entries = []
          for j in range(len(self.diffclass_subjectsfeat[i])):
              #Was chr(ch_class)
              ch_class = self.diffclass_subjectsfeat[i][j][-1] - 32
              subject_entries.append(ch_class)
          self.diffclass_subjectsclass.append(subject_entries)

      #Casting everything to an array. This will make it easier to manipulate data.
      for i in range(len(self.sameclass_subjectsfeat)):
          self.sameclass_subjectsfeat[i] = np.array(self.sameclass_subjectsfeat[i])

      for i in range(len(self.sameclass_subjectsclass)):
          self.sameclass_subjectsclass[i] = np.array(self.sameclass_subjectsclass[i])

      for i in range(len(self.diffclass_subjectsfeat)):
          self.diffclass_subjectsfeat[i] = np.array(self.diffclass_subjectsfeat[i])

      for i in range(len(self.diffclass_subjectsclass)):
          self.diffclass_subjectsclass[i] = np.array(self.diffclass_subjectsclass[i])

      #Instantiating important structures and setting important variables
      self.classifierObject = StaticClassification.MyStaticClassifiers()
      self.methodName = self.classifierObject.decisionTreeClassification
      self.folder = 'DecisionTreeClassifier'

  #Deleting all objects. Garbage collector already doing the job?
  def tearDown(self):
      pass

  #Logistic Regression
  def test_logisticRegression(self):
      model = self.myStaticClassifiers.logisticRegression(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())
  
  #Naive Bayes
  def test_naiveBayesClassifier(self):
      model = self.myStaticClassifiers.naiveBayesClassifier(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())
  
  #K Nearest Neighbors
  def test_kNearestNeighborsClassifier(self):
      model = self.myStaticClassifiers.kNearestNeighborsClassifier(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())

  #Radius Neighbours
  def test_radiusNeighborsClassification(self):
      model = self.myStaticClassifiers.radiusNeighborsClassification(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())

  #Decision Tree Classifier
  def test_decisionTreeClassification(self):
      model = self.myStaticClassifiers.decisionTreeClassification(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())

  #SVM Classifier
  def test_SVMClassifier(self):
      model = self.myStaticClassifiers.SVMClassifier(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())

  #Linear SVM Classifier
  def test_linearSVMClassifier(self):
      model = self.myStaticClassifiers.linearSVMClassifier(self.features, self.classes)
      predicted = model.predict(self.features)
      self.assertEqual(predicted.tolist(), self.classes.tolist())

  #Checking whether the types of the created data are OK
  def test_setUpTypes(self):
      self.assertTrue(type(self.sameclass_subjectsfeat) is list and
                      type(self.sameclass_subjectsfeat[0]) is np.ndarray and
                      type(self.sameclass_subjectsfeat[0][0]) is np.ndarray and
                      type(self.sameclass_subjectsclass) is list and
                      type(self.sameclass_subjectsclass[0]) is np.ndarray and
                      type(self.sameclass_subjectsclass[0][0]) is np.int64 and
                      type(self.diffclass_subjectsfeat) is list and
                      type(self.diffclass_subjectsfeat[0]) is np.ndarray and
                      type(self.diffclass_subjectsfeat[0][0]) is np.ndarray and
                      type(self.diffclass_subjectsclass) is list and
                      type(self.diffclass_subjectsclass[0]) is np.ndarray and
                      type(self.diffclass_subjectsclass[0][0]) is np.int64)

  def test_tenFold(self):
      pass

  #Testing subjectWiseCrossValidation by calling it twice, each time with a different input.
  #self.diffclass_* will test if the function can handle input with elements that have different sizes
  #self.sameclass_* will test if the function can handle input with elements that have same size
  #Several assertions are applied in this function. The first one checks if the returned data has the correct shape
  #The second one checks if *_statesList has the correct size
  #The third one checks if *_probabilitiesList has the correct size
  def test_subjectWiseCrossValidation(self):
      diff_statesList, diff_probabilitiesList = self.classifierObject.subjectWiseCrossValidation(self.methodName, 
                                                                                                 self.diffclass_subjectsfeat,
                                                                                                 self.diffclass_subjectsclass, self.folder)

      same_statesList, same_probabilitiesList = self.classifierObject.subjectWiseCrossValidation(self.methodName,
                                                                                                 self.sameclass_subjectsfeat,
                                                                                                 self.sameclass_subjectsclass, self.folder)

      self.assertTrue(type(diff_statesList) is list and
                      type(diff_statesList[0]) is np.ndarray and
                      type(diff_statesList[0][0]) is np.int64 and
                      type(diff_probabilitiesList) is list and
                      type(diff_probabilitiesList[0]) is np.ndarray and
                      type(diff_probabilitiesList[0][0]) is np.ndarray and
                      type(diff_probabilitiesList[0][0][0]) is np.float64 and
                      type(same_statesList) is list and
                      type(same_statesList[0]) is np.ndarray and
                      type(same_statesList[0][0]) is np.int64 and
                      type(same_probabilitiesList) is list and
                      type(same_probabilitiesList[0]) is np.ndarray and
                      type(same_probabilitiesList[0][0]) is np.ndarray and
                      type(same_probabilitiesList[0][0][0]) is np.float64)

      #Testing wheter the shape of statesList is correct
      diff_states_correct = True
      same_states_correct = True

      if len(diff_statesList) != len(self.diffclass_subjectsclass):
          diff_states_correct = False

      if len(same_statesList) != len(self.sameclass_subjectsclass):
          same_states_correct = False

      if diff_states_correct:
          for i in range(len(diff_statesList)):
              if len(diff_statesList[i]) != len(self.diffclass_subjectsclass[i]):
                  diff_states_correct = False
                  break

      if same_states_correct:
          for i in range(len(same_statesList)):
              if len(same_statesList[i]) != len(self.sameclass_subjectsclass[i]):
                  same_states_correct = False
                  break

      self.assertTrue(diff_states_correct)
      self.assertTrue(same_states_correct)

      #Testing wheter the shape of probabilitiesList is correct
      diff_probabilities_correct = True
      same_probabilities_correct = True

      if len(diff_probabilitiesList) != len(self.diffclass_subjectsfeat):
          diff_probabilities_correct = False

      if len(same_probabilitiesList) != len(self.sameclass_subjectsfeat):
          same_probabilities_correct = False

      if diff_probabilities_correct:
          for i in range(len(diff_probabilitiesList)):
              if len(diff_probabilitiesList[i]) != len(self.diffclass_subjectsfeat[i]):
                  diff_probabilities_correct = False
                  break

      if same_probabilities_correct:
          for i in range(len(same_probabilitiesList)):
              if len(same_probabilitiesList[i]) != len(self.sameclass_subjectsfeat[i]):
                  same_probabilities_correct = False
                  break

      self.assertTrue(diff_probabilities_correct)
      self.assertTrue(same_probabilities_correct)

if __name__ == '__main__':
    unittest.main()
