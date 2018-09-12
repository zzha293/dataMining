{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf200
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww12200\viewh12300\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 ReadMe\
//Heather Zhou and Bambi Zhang\
\
Before run the program, please ensure you have installed the below libraries:\
	pip install numpy\
	pip install sklearn\
	pip install json\
	pip install spicy\
	pip install re\
	pip install csv\
	pip install random\
\
Then using the following code to run:\
	1. Single classifier\
		a. SVM: \

\i\b 			python svm.py train.json test.json svmOutput.csv\

\i0\b0 		b. SGD: \
			
\i\b python sgd.py train.json test.json sgdOutput.csv
\i0\b0 \
		c. Random forest: \
			
\i\b python randomForest.py train.json test.json randFOutput.csv
\i0\b0 \
		d. Bayesian: \
			
\i\b python bayes.py train.json test.json bayesOutput.csv
\i0\b0 \
		e. Neural Network: \
			
\i\b python neuralnetwork.py train.json test.json nnOutput.csv
\i0\b0 \
		f. Logistic Regression: \
			
\i\b python logisticReg.py train.json test.json logReOutput.csv
\b0 \

\i0 \
\
	2. Stacking with three classifiers at the first layer\
		First we generate prediction files according to neural network, random forest, logistic regression and stochastic gradient four single classifiers.\

\i\b 			python stackingFirstLayer.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt\

\i0\b0 \
		Then we run the stacking based on \ul four\ulnone  of the basic single classifiers (neural network, random forest, logistic regression):\
		a. NN meta classifier\

\i\b 			python stackingbyNN.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4nn.csv\

\i0\b0 		b. Logistic Regression meta classifier\

\i\b 			python stackingbyreg.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4reg.csv\

\i0\b0 		c. SVM meta classifier\

\i\b 			python stackingbysvm.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4svm.csv\

\i0\b0 		d. SGD meta classifier \

\i\b 			python stackingbysgd.py train.json test.json out_randF.txt out_reg.txt out_nn.txt out_sgd.txt out_label.txt predict_4sgd.csv\
\
	\

\i0\b0 	3. Last, majority vote of all the previous outputs\
			
\i\b python major.py
\i0\b0 \
			(The output file is automatically saved as \'91majorOutput.csv\'92)
\i\b \
}