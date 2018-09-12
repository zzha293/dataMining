# encoding: utf-8
"Stacking methods that output the prediction of 4 base models of first layer"


import sys
import json
import re
import csv
import numpy as np
import scipy
import random
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# @param: fname<str>
# @return: ingredients<list<str>>
# @return: trainLabel<list<str>>
def dataReader_sgd(fname):

    "Function that read from a file and returns the ingredients in a recipe and a list of string with cuisine"

    with open(fname) as dataFile:
        data = json.load(dataFile)

    ingredients = []
    trainLabel = []
    for i in range(0, len(data)):
        trainLabel.append(data[i]['cuisine'].encode("ascii","replace"))
        recipe = ""
        for j in range(0, len(data[i]['ingredients'])):
            ingredientTemp = data[i]['ingredients'][j]
            ingredientTemp = removeSpecialCharacter(ingredientTemp)
            ingredientStr = ingredientTemp.encode("ascii","replace")
            ingredientStr = ingredientStr.replace(" ", "")
            recipe = recipe + ingredientStr + " "
        ingredients.append(recipe)

    return ingredients, trainLabel

# @param: fname<str>
# @return: ingredients<list<str>>
# @return: trainLabel<list<str>>
# @return: index<list<int>>
def dataReader(fname):

    "Function that read from a file and create list with all ingredients, a list of string with cuisine and list of index"

    with open(fname) as dataFile:
        data = json.load(dataFile)

    ingredients = set()
    trainLabel = []
    index = []
    for i in range(0, len(data)):
        trainLabel.append(data[i]['cuisine'])
        index.append(i)
        for j in range(0, len(data[i]['ingredients'])):
            ingredientTemp = data[i]['ingredients'][j]
            ingredientTemp = removeSpecialCharacter(ingredientTemp)
            ingredientStr = ingredientTemp.encode("ascii","replace")
            ingredients.add(ingredientStr)
    ingredients = list(ingredients)

    return ingredients, trainLabel, index

# @param: train<str>
# @param: ingredients<list<str>>
# @return: matrix<list<list<int>>>
# @return: ids<list<str>>
def createMatrix(train, ingredients):

    "Function that read from a file and create a matrix that labels 1 if the recipe contains an ingredient and a list of string with recipe ids"

    with open(train) as dataFile:
        data = json.load(dataFile)

    matrix = list()
    ids = list()
    for i in range(0, len(data)):
        ids.append(data[i]['id'])
        row = [0]*len(ingredients)
        for j in range(0, len(data[i]['ingredients'])):
            ingredientTemp = data[i]['ingredients'][j]
            ingredientTemp = removeSpecialCharacter(ingredientTemp)
            ingredientStr = ingredientTemp.encode("ascii","replace")
            for k in range(0, len(ingredients)):
                if ingredients[k] == ingredientStr:
                    row[k] = 1
        matrix.append(row)

    return matrix,ids

# @param: input<str>
# @return: input<str>
def removeSpecialCharacter(input):

    "Function that does simple clean up on ingredient strings"

    input = input.lower()
    while re.search(u'™', input, flags=re.U):
        match = re.search(u'™', input, flags=re.U)
        input = input[match.end():].lstrip()

    while re.search(u'®', input, flags=re.U):
        match = re.search(u'®', input, flags=re.U)
        input = input[match.end():].lstrip()
        # input = input[:match.start()] + input[match.end():]

    match = re.search(u'\)', input, flags=re.U)
    if match:
        if match.end() != len(input):
            input = input[match.end():].lstrip()
        else:
            m = re.search(u'\(', input, flags=re.U)
            input = input[:m.start()-1]
    input = re.sub(r'crushed|crumbles|ground|minced|powder|chopped|sliced', '', input)
    return input

# @param: trainMatrix<list<list<int>>>
# @param: trainLabel<list<str>>
# @param: predictM<list<list<int>>>
# @return: resultP<list<list<float>>>
def randomForest(trainMatrix, trainLabels, predictM):

    "An function that train and predict using random forest"

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(trainMatrix, trainLabels)

    resultP = clf.predict_proba(predictM)

    return resultP

# @param: trainMatrix<list<list<int>>>
# @param: trainLabel<list<str>>
# @param: predictM<list<list<int>>>
# @return: resultP<list<list<float>>>
def logisticReg(trainMatrix, trainLabels, predictM):

    "An function that train and predict using logistic regression"

    reg = linear_model.LogisticRegression()
    reg.fit(trainMatrix, trainLabels)

    resultP = reg.predict_proba(predictM)

    return resultP

# @param: trainMatrix<list<list<int>>>
# @param: trainLabel<list<str>>
# @param: predictM<list<list<int>>>
# @return: resultP<list<list<float>>>
def svm(trainMatrix, trainLabels, predictM):

    "An function that train and predict using Linear SVC (NOT APPLIED AS BASE MODEL)"

    lin_clf = LinearSVC(random_state=0)
    lin_clf.fit(trainMatrix, trainLabels)

    result = lin_clf.predict(predictM)

    return result

# @param: trainMatrix<list<list<int>>>
# @param: trainLabel<list<str>>
# @param: predictM<list<list<int>>>
# @return: resultP<list<list<float>>>
def nn(trainMatrix, trainLabels, predictM):

    "An function that train and predict using neural network"

    le = preprocessing.LabelEncoder()
    le.fit(trainLabels)
    trainLabels = le.transform(trainLabels)

    scaler = StandardScaler()
    scaler.fit(trainMatrix)
    trainMatrix = scaler.transform(trainMatrix)
    predictM = scaler.transform(predictM)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, alpha=1e-4,
                        solver='sgd', verbose=False, tol=1e-4, random_state=None,
                        learning_rate_init=.001)

    mlp.fit(trainMatrix, trainLabels)
    result = mlp.predict_proba(predictM)
    # result = le.inverse_transform(result)

    return result

# @param: sgdtrain<list<list<str>>>
# @param: trainl<list<str>>
# @param: sgdtest<list<list<str>>>
# @return: resultP<list<list<float>>>
def sgd(sgdtrain, trainl, sgdtest):

    "An function that train and predict using SGD"

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__loss': ('log', 'modified_huber'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(sgdtrain, trainl)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    result = grid_search.predict_proba(sgdtest)

    return result

# @param: matrix<list<list<int>>>
# @param: trainLabel<list<str>>
# @param: index<list<int>>
# @param: sgd_ingredient<list<list<str>>>
# @return: m1<list<str>>
# @return: m2<list<str>>
# @return: m3<list<str>>
# @return: m4<list<str>>
# @return: labels<list<str>>
def firstLayer(matrix, trainLabel, index, sgd_ingredient):

    "A function that randomly create 10 test folds. For each test fold, train each base models using the other 9 and predict the one. Returns the prediction of the four base model and the actual labels"

    random.shuffle(index)
    size = len(trainLabel)/10
    stackInput = []
    m1 = list()
    m2 = list()
    m3 = list()
    m4 = list()
    labels = list()
    prediction = list()

    for i in range(1, 11):
        train = []
        trainl = []
        test = []
        testl = []
        sgdtrain = []
        sgdtest = []
        if i == 10:
            beg = (i - 1) * size
            for j in range(0, len(index)):
                if j < beg:
                    train.append(matrix[j])
                    trainl.append(trainLabel[j])
                    sgdtrain.append(sgd_ingredient[j])
                else:
                    test.append(matrix[j])
                    testl.append(trainLabel[j])
                    sgdtest.append(sgd_ingredient[j])
        else:
            beg = (i - 1) * size
            end = i * size
            for j in range(0, len(index)):
                if j < beg or j >= end:
                    train.append(matrix[j])
                    trainl.append(trainLabel[j])
                    sgdtrain.append(sgd_ingredient[j])
                else:
                    test.append(matrix[j])
                    testl.append(trainLabel[j])
                    sgdtest.append(sgd_ingredient[j])
        if i == 1:
            m1 = randomForest(train, trainl, test)
            m2 = logisticReg(train, trainl, test)
            m3 = nn(train, trainl, test)
            m4 = sgd(sgdtrain, trainl, sgdtest)
            labels = testl
        else:
            m1 = np.concatenate((m1,randomForest(train, trainl, test)), axis=0)
            m2 = np.concatenate((m2,logisticReg(train, trainl, test)), axis=0)
            m3 = np.concatenate((m3,nn(train, trainl, test)), axis=0)
            m4 = np.concatenate((m4,sgd(sgdtrain, trainl, sgdtest)), axis=0)
            labels = labels + testl

    return m1, m2, m3, m4, labels

# @param: result<list<str>>
# @param: outfile<str>
def write(result, outfile):

    "Write the labels to file"

    myFile = open(outfile, "write")
    with myFile:
        for item in result:
            myFile.write("%s\n" % item)

    myFile.close()

# Process command line argument
train = sys.argv[1]
test = sys.argv[2]
output_randomF = sys.argv[3]
output_reg = sys.argv[4]
output_nn = sys.argv[5]
output_sgd = sys.argv[6]
output_label = sys.argv[7]

if train is not None:
    # get the training file
    allIngredients, trainLabel, index = dataReader(train)
    sgd_ingredient, sgd_label = dataReader_sgd(train)
    matrix, trainIDs = createMatrix(train, allIngredients)
else:
    print 'No training data, please re-enter\n'
    sys.exit('Program exit')

#Train the first layer base models
m1, m2, m3, m4, labels = firstLayer(matrix, trainLabel, index, sgd_ingredient)

#Write the output from first layer to files
np.savetxt(output_randomF, m1, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt(output_reg, m2, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt(output_nn, m3, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt(output_sgd, m4, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
write(labels, output_label)
