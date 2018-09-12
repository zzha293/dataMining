# dataMining
Before run the program, please ensure you have installed the below libraries:
	
	pip install numpy
	pip install sklearn
	pip install json
	pip install spicy
	pip install re
	pip install csv
	pip install random

Then using the following code to run:

	1. Single classifier
		a. SVM: 
			python svm.py train.json test.json svmOutput.csv
		b. SGD: 
			python sgd.py train.json test.json sgdOutput.csv
		c. Random forest: 
			python randomForest.py train.json test.json randFOutput.csv
		d. Bayesian: 
			python bayes.py train.json test.json bayesOutput.csv
		e. Neural Network: 
			python neuralnetwork.py train.json test.json nnOutput.csv
		f. Logistic Regression: 
			python logisticReg.py train.json test.json logReOutput.csv


	2. Stacking with three classifiers at the first layer
		First we generate prediction files according to neural network, random forest, logistic regression and stochastic gradient four single classifiers.
			python stackingFirstLayer.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt

		Then we run the stacking based on four of the basic single classifiers (neural network, random forest, logistic regression):
		a. NN meta classifier
			python stackingbyNN.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4nn.csv
		b. Logistic Regression meta classifier
			python stackingbyreg.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4reg.csv
		c. SVM meta classifier
			python stackingbysvm.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4svm.csv
		d. SGD meta classifier 
			python stackingbysgd.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4sgd.csv

	
	3. Last, majority vote of all the previous outputs
			python major.py
			(The output file is automatically saved as ‘majorOutput.csv’)
