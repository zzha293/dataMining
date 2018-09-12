# encoding: utf-8
"Stacking methods the prediction from SGD meta classifier"

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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# @param: fname<str>
# @return: ingredients<list<str>>
# @return: trainLabel<list<str>>
def dataReader_sgd(fname):

    "Function that read from a file and create a list of labels and a list of attributes, where all instances of a given categorical attributes are put into a list"

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

# @param: test<str>
# @return: testAttributee<list<str>>
# @return: ids<list<str>>
def sgdreadTest(test):

    "Function that read from a file and returns the ingredients in a recipe and a list of string with recipe ids"

    with open(test) as dataFile:
        data = json.load(dataFile)

    testAttribute = list()
    ids = list()
    for i in range(0, len(data)):
        ids.append(data[i]['id'])
        row = ""
        for j in range(0, len(data[i]['ingredients'])):
            ingredientTemp = data[i]['ingredients'][j]
            ingredientTemp = removeSpecialCharacter(ingredientTemp)
            ingredientStr = ingredientTemp.encode("ascii","replace")
            ingredientStr = ingredientStr.replace(" ", "")
            row = row + ingredientStr + " "
        testAttribute.append(row)

    return testAttribute,ids

# @param: m1<list<str>>
# @param: m2<list<str>>
# @param: m3<list<str>>
# @param: m4<list<str>>
# @return: prob_matrix<list<list<float>>>
def firstLayerReader(m1, m2, m3, m4):

    "A function that concatenates predicted probabilities from 4 base models into a input matrix for the second layer"

    proba_matrix = list()
    try:
        with open(m1) as m1, open(m2) as m2, open(m3) as m3, open(m4) as m4:
            a = m1.readlines()
            a = [x.strip('\n') for x in a]
            b = m2.readlines()
            b = [x.strip('\n') for x in b]
            c = m3.readlines()
            c = [x.strip('\n') for x in c]
            d = m4.readlines()
            d = [x.strip('\n') for x in d]

            for item1, item2, item3, item4 in zip(a, b, c, d):

                item1 = item1.split(" ")
                item1 = list(map(float, item1))
                item2 = item2.split(" ")
                item2 = list(map(float, item2))
                item3 = item3.split(" ")
                item3 = list(map(float, item3))
                item4 = item4.split(" ")
                item4 = list(map(float, item4))

                row = item1
                row.extend(item2)
                row.extend(item3)
                row.extend(item4)
                proba_matrix.append(row)
    except IOError as e:
        print 'Operation failed: %s' % e.strerror

    return proba_matrix

# @param: fname<str>
# @return: resultP<list<str>>
def labelReader(fname):

    "Read the file and parse it as a list of string"

    labels = list()
    with open(fname) as f:
        line = f.readlines()
        line = [str(x).replace("\n", "") for x in line]
        for x in line:
            labels.append(x)

    return labels

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

    "An function that train and predict using Linear SVC"

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

    return result

# @param: sgdtrain<list<list<str>>>
# @param: trainl<list<str>>
# @param: sgdtest<list<list<str>>>
# @return: resultP<list<list<float>>>
def sgd(sgdtrain, trainl, sgdtest):
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

# @param: outfile<str>
# @param: test<list<int>>
# @param: result<list<str>>
def write_to_file(outfile, test, result):

    "Write prediction to file"

    myFile = open(outfile, "write")
    with myFile:
        writer = csv.writer(myFile)
        myPredict = list()
        myPredict.append(['id', 'cuisine'])
        for id, predict in zip(test, result):
            row = list()
            row.append(id)
            row.append(predict)
            myPredict.append(row)
        writer.writerows(myPredict)

    myFile.close()

# Process command line argument
train = sys.argv[1]
test = sys.argv[2]
m1 = sys.argv[3]
m2 = sys.argv[4]
m3 = sys.argv[5]
m4 = sys.argv[6]
tlabel = sys.argv[7]
outFile = sys.argv[8]

if train is not None:
    # get the training file
    allIngredients, trainLabel, index = dataReader(train)
    sgd_ingredient, sgd_label = dataReader_sgd(train)
    matrix, trainIDs = createMatrix(train, allIngredients)
else:
    print 'No training data, please re-enter\n'
    sys.exit('Program exit')

if test is not None:
    ingredientMatrix, testIDs = createMatrix(test, allIngredients)
    sgdtest, sgdtestIDs = sgdreadTest(test)
else:
    print 'No test data, please re-enter\n'
    sys.exit('Program exit')

secondLayerInput = firstLayerReader(m1, m2, m3, m4)
labels = labelReader(tlabel)

#Train the grid search classifier
clf = SGDClassifier()
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'alpha': (0.00001, 0.000001),
    'penalty': ('l2', 'elasticnet', 'l1'),
    'n_iter': (10, 50),
}
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
grid_search.fit(secondLayerInput, labels)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#Train each base model from the whole training dataset, then make predictions on the testing data
prediction = list()
randomF_out = randomForest(matrix, trainLabel, ingredientMatrix)
reg_out = logisticReg(matrix, trainLabel, ingredientMatrix)
nn_out = nn(matrix, trainLabel, ingredientMatrix)
sgd_out = sgd(sgd_ingredient, sgd_label, sgdtest)

for item1, item2, item3, item4 in zip(randomF_out, reg_out, nn_out, sgd_out):
    row = np.append(item1, item2)
    row = np.append(row, item3)
    row = np.append(row, item4)
    prediction.append(row)

#Use prediction as input for the second layer stacking model
result = grid_search.predict(prediction)

write_to_file(outFile, testIDs, result)