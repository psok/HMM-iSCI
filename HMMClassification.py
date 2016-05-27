# Rewritten by Pichleap Sok (Jessie) <psok@luc.edu>

import numpy as np
import pandas as pd
from myHmm import GaussianHMM
from Util import *
from sklearn import cross_validation

class HMMClassification:
    """
    A simple class that helps in training, testing and cross-validating a HMM classifier.
    """
    
    #code taken from source - http://stackoverflow.com/questions/13219041/how-can-i-speed-up-transition-matrix-creation-in-numpy
    #Modified by @Author: Asma Mehjabeen  
    def getHMMProbabilities(self,markov_chain, activities):
        """
        Method that calculates the transition matrix and the initial state probabilities.
        
        Parameters
        ----------
        markov_chain:List of all the target activities of the subject (in the same order they were found in the file).
        activities: A unique list of activities containing all the activities that the user performed.
        
        Returns
        --------
        transition_matrix: An np matrix containing the transition probabilities
        initial_state_prob: An np array containing the initial state probabilities
        """
        
        '''Create a nxn dataframe to hold the transitions between activities. n represents the number of activities. This is 
        used for calculating the transition probabilities'''
        df = pd.DataFrame(index=activities, columns=activities)
        #Initially fill the data frame with 0 --> 0 probability of transitioning from one activity to the next.
        df = df.fillna(0)
        
        '''initialize the activity dictionary to hold activity count in target. This dictionary is used for holding the initial state
        probabilities'''
        state_dict ={}
        for activity in activities:
            state_dict[activity] = 0.0
            
        #If there is a transition from activity i to j then increment the count in the dataframe by 1.
        total_transitions = 0
        for i in range(1, len(markov_chain)):
            old_state = markov_chain[i - 1]
            new_state = markov_chain[i]
            #increment transition count in the dataframe
            df[old_state][new_state] += 1 
            #increment activity count in the dictionary
            state_dict[new_state] += 1.0
            total_transitions +=1 
        
        #Store all the activity count as a list. (Excludes activity labels)
        myList = list(state_dict.values())
        #Divide each number in the list by the total number of activities. (to get the probability of each activity).
        newList = [((x/total_transitions))/10000  for x in myList]
        initial_state_prob = newList
        
        '''This is done to make sure the sum of the initial state probability matrix is 1. 
        If not a small constant is added until the sum equals 1 (Constant value is always less than 0.1)'''
        startProbSum = sum(initial_state_prob)
        if startProbSum < 1.0:
            lastIndex = len(initial_state_prob) - 1
            remainder = 1.0
            for i in range(lastIndex):
                remainder -= initial_state_prob[i]
            initial_state_prob[lastIndex] = remainder
            
        initial_state_prob = np.array(initial_state_prob)
        '''Check if one of the probabilities is 0. If yes, normalize (This check is also done in the sklearn library. It's done here
        to avoid normalization of the matrix in the library code which would mess up the row sum again by a small constant and thus
        will throw and error and the program crashes)'''
        if not np.alltrue(initial_state_prob):
            normalize(initial_state_prob)
        
        '''Convert the data frame to an np.array. (A data frame was used initially just to add data to the correct activity label, since 
         we do not know the labels and their index as they are extracted dynamically from the subjects file'''
        transition_matrix = np.array(df.as_matrix(columns=None))
        
        # to avoid zero divided by zero when the transition matrix by their row sum
        for row in transition_matrix:        
            if np.all(row==0):            
                for i in range(len(row)):
                    row[i] = 1
        
        '''Divide all rows in the transition matrix by their row sum to get probabilities'''
        transition_matrix = np.mat((transition_matrix*1.0)/transition_matrix.sum(axis=1)[:,None])
        
        '''This is done to make sure the sum of the rows of the transition matrix is 1. 
        If not a small constant is added until the row sum equals 1 (Constant value is always less than 0.1)'''
        transition_matrix = transition_matrix.tolist()

        for row in transition_matrix:
            rowSum = sum(row)
            
            # To ensure no probability is smaller than 0.001
            maxIndex = np.argmax(row)
            for i in range(len(row)):
                if row[i] < 0.001 and i != maxIndex: 
                    row[i] = 0.001
                    row[maxIndex] = row[maxIndex] - row[i]
                
            if rowSum < 1.0:
                lastIndex = len(row) - 1
                remainder = 1.0
                for i in range(lastIndex):
                    remainder -= row[i]
                row[lastIndex] = remainder

        transition_matrix = np.mat(np.array(transition_matrix))
        
        '''Check if one of the probabilities is 0. If yes, normalize (This check is also done in the sklearn library. It's done here
        to avoid normalization of the matrix in the library code which would mess up the row sum again by a small constant and thus
        will throw and error and the program crashes)'''
        if not np.alltrue(transition_matrix):
            normalize(transition_matrix, axis=1)
            
#==============================================================================
        #Save transition_matrix and initial probabilities to file 
        #(One time only because transition matrix is consistent)
        filename = Constants.FINAL_RESULTS_FOLDER + 'HMMParameters.csv'
        df = pd.DataFrame(transition_matrix, index = activities, columns = activities)
        df.to_csv(filename)
        df = pd.DataFrame(initial_state_prob)
        df.to_csv(filename, index_col=False, header=False, mode = 'a')
#==============================================================================
        
        return transition_matrix,initial_state_prob
            
    def getMeans(self,features,target):
        """A simple method to get the mean values of the observations
        
        Parameters
        ----------
        features: Observation array of the HMM. shape = (n_samples,n_classes)
        target: Array of true activities of the subject. shape = (n_samples)
        
        Returns
        --------
        meansArray: array, shape (n_samples,n_classes) Mean parameters for each state/activity."""
        df = pd.DataFrame(features)
        df['target'] = target

        df = df.groupby(['target']).mean()
        
        filename = Constants.FINAL_RESULTS_FOLDER + 'HMMParameters.csv'
        df.to_csv(filename, index_col=False, header=False, mode = 'a')
        
        meansArray = np.array(df)
        return meansArray
        
        
    def trainHmmClassifier(self,train_features,train_target,transition_prob,initial_state_vector,meansArray):
        """A method to train the HMM model.
        
        Parameters
        ----------
        train_features: the observations of the HMM
        train_target: the true activities of the HMM
        
        Returns
        --------
        hmmModel: The trained HMM Model Classifier
        activities: The unique set of activities (list).
        
        """

        try: 
            #activities = np.unique(train_target) 
            #activities = np.array(['Lying', 'Sitting', 'StairClimbing', 'Standing', 'Walking', 'Wheeling'], dtype='<U13')
            #Get the transition and initial state probability matrices
            #transition_prob, initial_state_vector = self.getHMMProbabilities(train_target, activities)
            componentsCount = len(initial_state_vector)
            #Get the means Array.
            #meansArray = self.getMeans(train_features,train_target)
            #Initialize the model's constructor
            hmmModel = GaussianHMM(n_components = componentsCount, covariance_type = "diag", startprob = initial_state_vector, transmat = transition_prob)
            
            #Train the model by fitting the data. Must always fit the obs data before change means and covars
            hmmModel.fit([train_features])
            
            hmmModel.means_ = meansArray
            hmmModel.covars_ = 0.05*np.ones((componentsCount,componentsCount), dtype=float)
            
        except ValueError as e:
            '''If there is a value error as a result of the the row sum of transition matrix or initial state probability not equating to 1,
            call the method again. But due to a minor fix in the library code this exception handling is not needed anymore'''
            print(e)
            print("Value error in training. Calling the method again")
            return self.trainHmmClassifier(train_features,train_target,transition_prob,initial_state_vector,meansArray)
            
        else:
            return hmmModel
            
            
    
    def testHmmClassifier(self,hmmModel,test_features,test_target,activities):
        """
        A method to test the HMM classifier
        
        Parameters
        ----------
        hmmModel : A trained HMM classifier model
        test_features: The observations used for testing
        test_target: The true activities used for validating the model
        activities: The unique set of activities
        
        Returns
        --------
        score: The accuracy of the HMM model as a percent. (float)
        hmmStates: List of the states as predicted by the HMM model based on the observations given.
        """
        #Use of LR probability to predict the states. Returns the index of the states.
        hmmStates = hmmModel.predict(test_features)
        hmmStates = np.array(hmmStates).tolist()
        #Convert state index back to strings
        hmmStates = [activities[x] for x in hmmStates]
            
        #Get the probability of success HMM
        score = (self.comp(hmmStates,test_target))*100
        return score,hmmStates 
        
    #Just a function to compare the lists.
    # @Author: Ranulfo Neto
    def comp(self,A, B):
        mySum = 0
        for x in range(len(B)):
            if A[x] == B[x]:
                mySum += 1
        return (mySum/float(len(B)))
        
    def hmmNFoldValidation(self,targetMaster,probabilitiesMaster, filePath):

        listLength = len(targetMaster)
        
        HMMScoresList = []
        HMMStatesList = []
        newTargetList = []
        
        activities = np.array(['Lying', 'Sitting', 'StairClimbing', 'Standing', 'Walking', 'Wheeling'], dtype='<U13')
        markov_chain = [item for sublist in targetMaster for item in sublist]   #just a flatten list of targetMaster      
        transition_prob, initial_state_vector = self.getHMMProbabilities(markov_chain, activities)
        train_features = [item for sublist in probabilitiesMaster for item in sublist]
        train_target = [item for sublist in targetMaster for item in sublist]
        meansArray = self.getMeans(train_features,train_target)
        
        accuracy_str = ""
        
        for i in range(listLength):
            train = np.array(probabilitiesMaster[i])
            target = np.array(targetMaster[i])
            
            kf_total = cross_validation.KFold(len(train), n_folds=20, shuffle=False, random_state=6)
        
            scores = []
            predicted = []
            newTarget = []
            for train_index, test_index in kf_total:
                train_prob = train[train_index]
                test_prob = train[test_index]
                train_target = target[train_index]
                test_target = target[test_index]  
                hmmModel = self.trainHmmClassifier(train_prob,train_target,transition_prob,initial_state_vector,meansArray)
                cvScore,hmmstates = self.testHmmClassifier(hmmModel,test_prob,test_target,activities)
                scores.append(cvScore)
                predicted.append(hmmstates)
                newTarget.append(test_target)
        
            score = np.mean(scores)
            predicted = [item for sublist in predicted for item in sublist]
            newTarget = [item for sublist in newTarget for item in sublist]
    
            HMMScoresList.append(score)
            HMMStatesList.append(predicted)
            newTargetList.append(newTarget)
            print(('HMM 20-fold Accuracy Subject_' + str(i+1) + ' = ' + str(score) + '%'))
            accuracy_str += 'HMM 20-fold Accuracy Subject_' + str(i+1) + ' = ' + str(score) + '%' + '\n'
            
        print(('Overall HMM 20-fold Accuracy = ' + str(np.mean(HMMScoresList)) + '%'))
        accuracy_str += 'Overall HMM 20-fold Accuracy = ' + str(np.mean(HMMScoresList)) + '%'
        
        # Write accuracy of each subject to file
        filename = filePath + Constants.HMMFolder + '_20Fold_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)
            
        return HMMStatesList,newTargetList
                        
    def hmmClassifierSubjectWiseValidation(self,targetMaster,probabilitiesMaster, filePath):
        """A method written to cross validate the HMM classifier using subject wise cross validation.
         
        Parameters
        ----------
        probabilitiesList: A list containing the probabilistic estimates of all the subjects. Each of the element in the list is an 
        observations array of shape [n_samples,n_features] of a particular subject. Size of list = Number of subjects.
        
        targetMaster: A list containing the array of target activities of all the subjects. Each of the element in the list is a
        target array of shape [n_samples] of a particular subject. Size of list = Number of subjects.
        
        Results
        -------
        HMMStatesList: A list of lists where each sub-list contains the predicted HMM States of a particular subject(the one 
        that is left out). Size of list = Number of subjects.  
        """

        listLength = len(targetMaster)
        HMMScores = []
        HMMStatesList = []
        
        activities = np.array(['Lying', 'Sitting', 'StairClimbing', 'Standing', 'Walking', 'Wheeling'], dtype='<U13')
        markov_chain = [item for sublist in targetMaster for item in sublist]   #just a flatten list of targetMaster      
        transition_prob, initial_state_vector = self.getHMMProbabilities(markov_chain, activities)
        train_features = [item for sublist in probabilitiesMaster for item in sublist]
        train_target = [item for sublist in targetMaster for item in sublist]        
        meansArray = self.getMeans(train_features,train_target)
        
        '''for loop that aggregates the features are targets of all but one subject into train_features and train_target arrays respectively.
        The features and target activities of the remaining subject are in test_features and test_target arrays respectively. The model is
        then trained with the train_features and train_target arrays and tested with the test_features and test_target arrays.
        '''
        
        accuracy_str = ""
        for i in range(listLength):
            train_lrProb = []
            train_target = []
            test_lrProb = []
            test_target = []
            for k in range(listLength):
                if(k == i):
                    test_lrProb = probabilitiesMaster[k]
                    test_target = targetMaster[k]
                else:
                    #training data                
                    lrProb = probabilitiesMaster[k]
                    #print  ("Shape: k= "+str(k)+" "+str(lrProb.shape))
                    train_lrProb += lrProb.tolist()
                    target = targetMaster[k].tolist()
                    train_target += target

            train_lrProb = np.array(train_lrProb)
            #print  ("Shape: i= "+str(i)+" "+str(train_lrProb.shape))
            train_target = np.array(train_target)   
            
            #Train the HMM classifier
            hmmModel = self.trainHmmClassifier(train_lrProb,train_target,transition_prob,initial_state_vector,meansArray)

            #Test the HMM classifier
            score,hmmStates = self.testHmmClassifier(hmmModel,test_lrProb,test_target,activities)
            
            HMMScores.append(score)  
            HMMStatesList.append(hmmStates)
            #print the accuracy
            print(('HMM Subject-Wise Accuracy Subject_'+str(i+1)+' = '+str(score)+'%'))
            accuracy_str += 'HMM Subject-Wise Accuracy Subject_'+str(i+1)+' = '+str(score)+'%' + '\n'

        print(('Overall HMM Subject-Wise Validation Accuracy = '+str(np.mean(HMMScores))+'%'))
        accuracy_str += 'Overall HMM Subject-Wise Validation Accuracy = '+str(np.mean(HMMScores))+'%'
        
        # Write accuracy of each subject to file
        filename = filePath + Constants.HMMFolder + '_SubjectWise_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)

        return HMMStatesList

    def hmmWithinSubjectWiseValidation(self,targetMaster,probabilitiesMaster, filePath):
        
        listLength = len(targetMaster)
        
        HMMScoresList = []
        HMMStatesList = []
        newTargetList = []
        
        activities = np.array(['Lying', 'Sitting', 'StairClimbing', 'Standing', 'Walking', 'Wheeling'], dtype='<U13')
        markov_chain = [item for sublist in targetMaster for item in sublist]   #just a flatten list of targetMaster      
        transition_prob, initial_state_vector = self.getHMMProbabilities(markov_chain, activities)
        train_features = [item for sublist in probabilitiesMaster for item in sublist]
        train_target = [item for sublist in targetMaster for item in sublist]
        meansArray = self.getMeans(train_features,train_target)
        
        accuracy_str = ""
        
        for i in range(listLength):
            train = np.array(probabilitiesMaster[i])
            target = np.array(targetMaster[i])
            
            kf_total = cross_validation.KFold(len(train), n_folds=10, shuffle=False, random_state=6)
        
            scores = []
            predicted = []
            newTarget = []
            for train_index, test_index in kf_total:
                train_prob = []
                test_prob = []
                train_target = []
                test_target = []
                
                for k in range(listLength):
                    if(k == i):
                        test_prob = train[test_index]
                        test_target = target[test_index]  
                        train_prob += train[train_index].tolist()
                        train_target += target[train_index].tolist()
                        
                    else:
                        train_prob += probabilitiesMaster[k].tolist() 
                        train_target += targetMaster[k].tolist()
                        
                hmmModel = self.trainHmmClassifier(train_prob,train_target,transition_prob,initial_state_vector,meansArray)
                cvScore,hmmstates = self.testHmmClassifier(hmmModel,test_prob,test_target,activities)
                scores.append(cvScore)
                predicted.append(hmmstates)
                newTarget.append(test_target)
        
            score = np.mean(scores)
            predicted = [item for sublist in predicted for item in sublist]
            newTarget = [item for sublist in newTarget for item in sublist]
    
            HMMScoresList.append(score)
            HMMStatesList.append(predicted)
            newTargetList.append(newTarget)
            print(('HMM Within-Subjectwise Accuracy Subject_' + str(i+1) + ' = ' + str(score) + '%'))
            accuracy_str += 'HMM Within-Subjectwise Accuracy Subject_' + str(i+1) + ' = ' + str(score) + '%' + '\n'
            
        print(('Overall HMM Within-Subjectwise Accuracy = ' + str(np.mean(HMMScoresList)) + '%'))
        accuracy_str += 'Overall HMM Within-Subjectwise Accuracy = ' + str(np.mean(HMMScoresList)) + '%'
        
        # Write accuracy of each subject to file
        filename = filePath + Constants.HMMFolder + '_WithinSubjectwise_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)
            
        return HMMStatesList,newTargetList
        
    def HmmPerSubject(self,targetMaster,probabilitiesList):
        """A method written to train the model on a single subjects data and test it on the same. No cross-validation done.
        
        Parameters
        ----------
        probabilitiesList: A list containing the probabilistic estimates of all the subjects. Each of the element in the list is an 
        observations array of shape [n_samples,n_features] of a particular subject. Size of list = Number of subjects.
        
        targetMaster: A list containing the array of target activities of all the subjects. Each of the element in the list is a
        target array of shape [n_samples] of a particular subject. Size of list = Number of subjects.
        
        Results
        -------
        HMMStates: A list of lists where each sub-list contains the predicted HMM States of a particular subject. Size of list = Number of subjects.  
        """
        HMMScores = []
        HMMStates = []
        for i in range(len(targetMaster)):
            target = targetMaster[i]
            probabilities = probabilitiesList[i]
            hmmModel,activities = self.trainHmmClassifier(probabilities,target)
            score,hmmStates = self.testHmmClassifier(hmmModel,probabilities,target,activities)
            HMMScores.append(score)
            HMMStates.append(hmmStates)         
            print(('Hybrid Classifier Accuracy Subject_'+str(i+1)+' = '+str(score)+'%'))        
        print(('Hybrid Classifier Accuracy = '+str(np.mean(HMMScores))+'%'))
        return HMMStates
        
    def adjustProbabilities(self,probabilitiesMaster,targetMaster):
        """Method to standardize the data by adding missing activities so that subject wise cross validation can be done.
        It finds the largest set of activities from among the users. For example: Subject 13 is currently doing only 4 activities and Subject 1 does 5.
        So it adds a column of observations or probabilistic estimates for the missing activities in Subject 13's probabilities array and then
        normalizes it to get rid of the 0 probabilities."""
        #Get the subjects data that has max number of activities.
        myMax = 0
        for i in range(len(targetMaster)):
            temp = len(np.unique(targetMaster[i]))
            #temp = len(max(x,key = len)) 
            if(temp > myMax):
                myMax = temp
                #myList = x
                listIndex = i
        
        #listIndex = probabilitiesMaster.index(myList)
        activities = np.unique(targetMaster[listIndex])
        print("Activities")
        print(activities)

        print("targetMaster")
        print((targetMaster[listIndex]))
    
        #print activities
        for i in range(len(probabilitiesMaster)):
            if not i==listIndex:
                columnAdded = False
                subjectsActivities = np.unique(targetMaster[i])
                #print ("Subject :"+str(i+1))
                print("Activities")
                print(activities)
                print("")
                print("Subject Activities")
                print(subjectsActivities)

                if(myMax != len(subjectsActivities)):                
                    for k in range(myMax):
                        #print ("Index :"+str(k))
                        if k == len(subjectsActivities):
                            #print (subjectsActivities[k],activities[k])
                            subjectsActivities = np.insert(subjectsActivities,k,activities[k],axis = 0)
                            probabilitiesMaster[i] = np.insert(probabilitiesMaster[i],k,0,axis = 1)
                            #print probabilitiesMaster[i]
                            columnAdded = True   

                        elif (subjectsActivities[k] != activities[k]):
                            #print (subjectsActivities[k],activities[k])
                            subjectsActivities = np.insert(subjectsActivities,k,activities[k],axis = 0)
                            probabilitiesMaster[i] = np.insert(probabilitiesMaster[i],k,0,axis = 1)
                            #print probabilitiesMaster[i]
                            columnAdded = True

                    if(columnAdded == True):
                        normalize(probabilitiesMaster[i], axis=1)
                        #print "Column added"