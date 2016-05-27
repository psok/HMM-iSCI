# Rewritten by Pichleap Sok (Jessie) <psok@luc.edu>

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from Util import *
from Constants import *
from sklearn.grid_search import GridSearchCV


class MyStaticClassifiers:
    """A simple class that has methods to train and cross validate  different classifiers such as
        -the Regularized Logistic Regression Classifier
        -the Naive Bayes Classifier
        -k-Nearest Neighbor Classifier
        -Decision Trees Classifier
        -Support Vector Machines Classifier
        -LinearSVM (does not give probabilities)
        -Radius Neighbor Classifier (does not work with the iSCI patient data set)"""
    
    
    def randomForestClassifier(self, X_train, y_train):
        tuned_parameters = [{'n_estimators': [1, 10, 100]}]
        model = GridSearchCV(RandomForestClassifier(n_estimators=10), tuned_parameters, cv=5) 
        model.fit(X_train, y_train)
        print(model.best_estimator_)
        return model
        
    def logisticRegression(self,X_train,y_train):
        """Method to train a regularized logistic regression classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained Logistic Regression model."""
        
        tuned_parameters = [{'C': [1e-2, 0.1, 1, 10, 100, 1000], 'penalty':['l1']}]
        model = GridSearchCV(linear_model.LogisticRegression(C=1), tuned_parameters, cv=5) 
        model.fit(X_train,y_train)
        print(model.best_estimator_)
        return model
        
    def NBMeanAndVariance(self,X,y):
        df = pd.DataFrame(X)
        df['target'] = y
        df1 = df.groupby(['target']).mean()
        meansArray = np.array(df1)
        df = df.groupby(['target']).var()
        varianceArray = np.array(df)
        return meansArray, varianceArray
        
        
    def naiveBayesClassifier(self,X_train,y_train):
        """Method to train a regularized logistic regression classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained Logistic Regression model."""
        
        meansArray, varianceArray = self.NBMeanAndVariance(X_train,y_train)
        
        model = GaussianNB()
        #fit the model with training data
        model.fit(X_train,y_train)
        model.theta_ = meansArray
        model.sigma_ = varianceArray
        
        return model
    
    def kNearestNeighborsClassifier(self, X_train,y_train):
        """Method to train a k nearest neighbor classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained nearest neighbor model."""
        
        tuned_parameters = [{'n_neighbors': [1, 3, 5, 20]}]
        model = GridSearchCV(KNeighborsClassifier(n_neighbors=1), tuned_parameters, cv=5) 
        model.fit(X_train,y_train)
        
        return model
     
    # Cannot be applied to the data in question since not enough neighbors are found in a given radius.      
    def radiusNeighborsClassification(self,X_train,y_train):
        """Method to train a radius nearest neighbor classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained nearest neighbor model."""
        
        #Initialize the constructor
        model = RadiusNeighborsClassifier(radius=1.0, weights = 'uniform' )
        #fit the model with training data
        model.fit(X_train,y_train)
        
        return model


    def decisionTreeClassification(self, X_train,y_train):
        """Method to train a decision tree classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained nearest neighbor model."""
        
        tuned_parameters = [{'min_samples_split': [1, 5, 10, 100]}]
        model = GridSearchCV(DecisionTreeClassifier(min_samples_split=1), tuned_parameters, cv=5)
        model.fit(X_train,y_train)
        
        return model
        
        
    def SVMClassifier(self, X_train,y_train):
        """Method to train a support vector machine classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained nearest neighbor model."""
        
        C = [1e-2, 0.1, 1, 10, 100, 1000]
        gamma = [1e-3] #[1e-3, 1e-4]
        tuned_parameters = [{'kernel': ['linear'], 'C': C},
                            {'kernel': ['rbf'], 'gamma': gamma,'C': C}]
                    
        model = GridSearchCV(svm.SVC(C=1.0, probability=True), tuned_parameters, cv=5)   
        model.fit(X_train,y_train)
        
        return model
   
    # Cannot be applied when using this in conjunction with dynamic estimation since this does not give the probability
    #estimates of the states for each sample. But, in generel it predicts state labels better than SVC with a linear kernel!    
    def linearSVMClassifier(self, X_train,y_train):
        """Method to train a radius nearest neighbor classifier
        
        Parameters
        ----------
        X_train: Array shape [n_samples,n_features] for training the model with features
        y_train: Array shape[n_samples] for training the model with features of that target
        
        Returns
        ---------
        model: The trained nearest neighbor model."""
        
        #Initialize the constructor
        model = svm.LinearSVC()
        #fit the model with training data
        model.fit(X_train,y_train)
        return model
        
    def nFoldValidationOnSubjects(self,func, featuresMaster,targetMaster,filePath):
        """A method to cross validate the classifier using 20-fold cross validation
        
        Parameters
        ----------
        featuresMaster: A list containing the features of all the subjects. Each of the element in the list is a
        features array of shape [n_samples,n_features] of a particular subject. Size of list = Number of subjects.
        
        targetMaster: A list containing the array of target activities of all the subjects. Each of the element in the list is a
        target array of shape [n_samples] of a particular subject. Size of list = Number of subjects.
        
        func: The classification function to be called
        
        filePath: The location where the results need to be saved. 
        
        Returns
        --------
        predictedStatesList: List that holds the predicted states array of all the subjects. Each element in the list is an array
        of the shape [n_samples] where each row is a predicted state of a particular subject. Size of list = Number of subjects.
        
        probabilitiesList: List that holds the predicted state probabilities of all the subjects. Each element is
        an array of the shape [n_samples,n_classes] containing the predicted probabilities of one particular subject. Size of list = Number of subjects."""
            
        listLength = len(targetMaster)
        #List that holds the leave one subject out cross validated scores.
        scores = []
        probabilitiesList = []
        predictedList = []
        newTargetList = []
        samples = 0
        
        hyperParameters = ""
        accuracy_str = ""
    
        for i in range(listLength):
            samples += len(featuresMaster[i])
            features = np.array(featuresMaster[i])
            target = np.array(targetMaster[i])  
            model = func(features,target)
            
            #store hyperparameters of all classiifers except for naiveBayesClassifier as it has no hyperparameter
            if(func.__name__ != 'naiveBayesClassifier'):
                hyperParameters += "Subject_" + str(i+1) + "\n"
                hyperParameters += str(model.best_estimator_) + "\n"
            
            predicted, probas, cvScore, newTarget = nFoldCrossValidator(model,features,target)
                
            scores.append(cvScore)
            probabilitiesList.append(probas)
            predictedList.append(predicted)
            newTargetList.append(newTarget)
            
            print(('20-Fold Accuracy Subject_' + str(i+1) + ' = ' + str(cvScore)))
            accuracy_str += '20-Fold Accuracy Subject_' + str(i+1) + ' = ' + str(cvScore) + '\n'
            
            #Write stateProbabilities, modelStates and target activities of each subject to a .csv file.
            filename = filePath + 'Subject_'+ str(i+1) +'.csv'
            df = pd.DataFrame(np.array(probas))
            df['modelStates'] = np.array(predicted)
            df['target'] = np.array(newTarget)
            df.to_csv(filename)
        
        scores = np.array(scores)
        print(('Overall 20-fold Accuracy = '+str(np.mean(scores))+'%'))
        accuracy_str += 'Overall 20-fold Accuracy = '+str(np.mean(scores))+'%'
        
        #Write hyperparameters to file
        filename = filePath + "Hyperparameters.txt"
        with open(filename, "w") as textfile:
            textfile.write(hyperParameters)
        
        #Write accuracy of each subject to file
        filename = filePath + Constants.StaticFolder + '_20Fold_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)
            
        print('Total samples = ' + str(samples))
        return predictedList,probabilitiesList,newTargetList


    def subjectWiseCrossValidation(self, func, featuresMaster, targetMaster , filePath):
        """A method to cross validate the classifier using leave one subject out cross validation
        
        Parameters
        ----------
        featuresMaster: A list containing the features of all the subjects. Each of the element in the list is a
        features array of shape [n_samples,n_features] of a particular subject. Size of list = Number of subjects.
        
        targetMaster: A list containing the array of target activities of all the subjects. Each of the element in the list is a
        target array of shape [n_samples] of a particular subject. Size of list = Number of subjects.
        
        func: The classification function to be called
        
        Returns
        --------
        predictedStatesList: List that holds the predicted states array of all the subjects. Each element in the list is an array
        of the shape [n_samples] where each row is a predicted state of a particular subject. Size of list = Number of subjects.
        
        probabilitiesList: List that holds the predicted state probabilities of all the subjects. Each element is
        an array of the shape [n_samples,n_classes] containing the predicted probabilities of one particular subject. Size of list = Number of subjects."""
        
        listLength = len(targetMaster)
        #List that holds the leave one subject out cross validated scores.
        scores = []
        probabilitiesList = []
        predictedList = []
        
        '''for loop that aggregates the features are targets of all but one subject into train_features and train_target arrays respectively.
        The features and target activities of the remaining subject are in test_features and test_target arrays respectively. The model is
        then trained with the train_features and train_target arrays and tested with the test_features and test_target arrays.
        '''

        hyperParameters = ""
        accuracy_str = ""
        for i in range(listLength):
            train_features = []
            train_target = []
            test_features = []
            test_target = []
            for k in range(listLength):
                if(k == i):
                    test_features = featuresMaster[k]
                    test_target = targetMaster[k]     
              
                else:
                    features = (featuresMaster[k]).tolist()
                    train_features += features 
                    target = targetMaster[k].tolist()
                    train_target += target
            train_features = np.array(train_features)
            train_target = np.array(train_target)

            model = func(train_features,train_target)
            
            #store hyperparameters of all classiifers except for naiveBayesClassifier as it has no hyperparameter
            if(func.__name__ != 'naiveBayesClassifier'):
                hyperParameters += "Subject_" + str(i+1) + "\n"
                hyperParameters += str(model.best_estimator_) + "\n"

            #Get the probability of each state from the prediction of model
            stateProbabilities = model.predict_proba(test_features)
            
            probabilitiesList.append(stateProbabilities)
            
            #Get the predicted states of the model and add to the master list
            states = model.predict(test_features)
            predictedList.append(states)
            
            #Get the probability of success of the model
            modelScore = (model.score(test_features,test_target))*100.0
            scores.append(modelScore)
            print(('Subject-Wise Accuracy Subject_' + str(i+1) + ' = ' + str(modelScore)))
            accuracy_str += 'Subject-Wise Accuracy Subject_' + str(i+1) + ' = ' + str(modelScore) + '\n'
            
            #Write stateProbabilities, modelStates and target activities of each subject to a .csv file.
            filename = filePath + 'Subject_' + str(i+1) + '.csv'      
            df = pd.DataFrame(stateProbabilities)
            df['modelStates'] = states
            df['target'] = test_target
            df.to_csv(filename)
    
        print(('Overall Subject-Wise Accuracy = '+str(np.mean(scores))+'%'))
        accuracy_str += 'Overall Subject-Wise Accuracy = '+str(np.mean(scores))+'%'
        
        # Write hyperparameters to file
        filename = filePath + "Hyperparameters.txt"
        with open(filename, "w") as textfile:
            textfile.write(hyperParameters)
            
        # Write accuracy of each subject to file
        filename = filePath + Constants.StaticFolder + '_SubjectWise_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)
        
        return predictedList,probabilitiesList
        
    def withinSubjectWiseValidation(self,func, featuresMaster,targetMaster,filePath):
        
        ''' A hybrid cross validation between within subject and subject wise. 
        This type of cross validation is very expensive. Improvement is needed. 
        This method could potentially improve the accuracy over within-subject cross validation
        because for now we don't have enough data to train each individual. 
        Therefore, by training on across-subjects and test on each fold of the individual
        the accuracy can be better.
        '''
        listLength = len(targetMaster)
        #List that holds the leave one subject out cross validated scores.
        scoresList = []
        probabilitiesList = []
        predictedList = []
        newTargetList = []
        samples = 0
        
        hyperParameters = ""
        accuracy_str = ""
    
        for i in range(listLength):
            
            samples += len(featuresMaster[i])
            features = np.array(featuresMaster[i])
            target = np.array(targetMaster[i])  
            model = func(features,target)
            if(func.__name__ != 'naiveBayesClassifier'):
                hyperParameters += "Subject_" + str(i+1) + "\n"
                hyperParameters += str(model.best_estimator_) + "\n"
                
            
            #Get the probability of success LR with cross validation.
            kf_total = cross_validation.KFold(len(features), n_folds=10, shuffle=False, random_state=6)
            probabilities = []
            predictions = []
            newTarget = []
            scores = []
            for train_index, test_index in kf_total:
                train_feature = []
                train_target = []
                test_feature = []
                test_target = []
                for k in range(listLength):
                    if(k == i):
                        test_feature = features[test_index]
                        test_target = target[test_index]
                        train_feature += features[train_index].tolist()
                        train_target += target[train_index].tolist()
                        
                    else:
                        train_feature += featuresMaster[k].tolist() 
                        train_target += targetMaster[k].tolist()
                        
                model.fit(train_feature, train_target)
                probas = model.predict_proba(test_feature)
                probabilities.append(probas)
                cvPredict = model.predict(test_feature)
                predictions.append(cvPredict)
                cvScore = model.score(test_feature,test_target)
                scores.append(cvScore)
                newTarget.append(test_target)
            
            predictions = [item for sublist in predictions for item in sublist]
            probabilities = [item for sublist in probabilities for item in sublist]
            newTarget = [item for sublist in newTarget for item in sublist]
                        
            scores = np.mean(scores)*100
            scoresList.append(scores)
            probabilitiesList.append(probabilities)
            predictedList.append(predictions)
            newTargetList.append(newTarget)
            
            print(('Within-Subjectwise Accuracy Subject_' + str(i+1) + ' = ' + str(scores)))
            accuracy_str += 'Within-Subjectwise Accuracy Subject_' + str(i+1) + ' = ' + str(scores) + '\n'
            
            #Write stateProbabilities, modelStates and target activities of each subject to a .csv file.
            filename = filePath + 'Subject_'+ str(i+1) +'.csv'
            df = pd.DataFrame(np.array(probabilities))
            df['modelStates'] = np.array(predictions)
            df['target'] = np.array(newTarget)
            df.to_csv(filename)
        
        overall_score = np.mean(np.array(scoresList))
        print(('Overall Within-Subjectwise Accuracy = '+str(overall_score)+'%'))
        accuracy_str += 'Overall Within-Subjectwise Accuracy = '+str(overall_score)+'%'
        
        #Write hyperparameters to file
        filename = filePath + "Hyperparameters.txt"
        with open(filename, "w") as textfile:
            textfile.write(hyperParameters)
        
        #Write accuracy of each subject to file
        filename = filePath + Constants.StaticFolder + '_WithinSubjectwise_Accuracy.txt'
        with open(filename, "w") as textfile:
            textfile.write(accuracy_str)
            
        print('Total samples = ' + str(samples))
        return predictedList,probabilitiesList,newTargetList
