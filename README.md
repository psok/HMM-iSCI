# HMM-iSCI

<strong>Requirements:</strong>

This project runs on python 3.4.4. <br /> 
You can run with either Spyder or Jupyter Notebook. <br />
One of the easiest ways is to install Anaconda package 4.0.5. which includes both Spyder and Jupyter Notebook.

<strong>Installation:</strong>

1. Download the project folder <br />
2. Unzip the folder <br /> 
3. Import the project to Spyder or Jupyter Notebook <br />
4. Run <strong>PreprocessData.py</strong> file to preprocess the raw data for each subject and store into <strong>*DataFrames*</strong> <br />
5. Run <strong>GetFeaturesTarget.py</strong> file to extract 2-second clips and extract features from the 2-second clips. Those features will then be written to files and stored into <strong>*FeaturesTarget*</strong> folder <br />
6. Run the main file <strong>HmmCode.py</strong> to see the results<br />

Notes: Step 4 and 5 are not neccessarily repeated every time you want to run the code. If there is nothing changed to the files, you can skip step 4 and 5. 

Without making any modification, the code will automatically run all classifiers, including SVMClassifier, logisticRegression, decisionTreeClassification, kNearestNeighborsClassifier, naiveBayesClassifier, and randomForestClassifier <br />
For each classifier, three cross validation methods will run: within-subject, across-subject and hybrid cross validation (haven't tested yet). <br />
Results of all classifiers will be stored in <strong>*Results/ClassifierName*</strong> folder, where ClassifierName is the name of the classifier used. 

<strong>*FeaturesTarget*</strong> folder contains csv files of features and target of all the thirdteen subjects

<strong>To change the clip size:</strong>

1. Open <strong>GetFeaturesTarget.py</strong> file into Spyder or Jupyter Notebook <br /> 
2. Search for #WHERE TO CHANGE CLIP SIZE comment <br /> 
3. Change the clip size <br />
4. Run the file <br />

After running the file, <strong>*FeaturesTarget*</strong> folder will be overwritten. 
