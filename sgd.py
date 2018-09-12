# encoding: utf-8
"Single classifier using stochastic gradient descent"


import sys
import json
import re
import csv
import numpy
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# @param: fname<str>
# @return: ingredients<list<str>>
# @return: trainLabel<list<str>>
def dataReader(fname):

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
# @return: testAttributee<list<str>>
# @return: ids<list<str>>
def readTest(test):

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

    return input

# @param: outfile<str>
# @param: test<list<int>>
# @param: result<list<str>>
def write_to_file(outfile, test, result):

    "Write prediction to file"

    myFile = open(output, "write")
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
output = sys.argv[3]
if train is not None:
    # get the training file
    ingredients, trainLabel = dataReader(train)
else:
    print 'No training data, please re-enter\n'
    sys.exit('Program exit')

if test is not None:
    ingredientMatrix, testIDs = readTest(test)
else:
    print 'No test data, please re-enter\n'
    sys.exit('Program exit')

if output is not None:
    # write into output file
    outFile = output
else:
    print 'No output file specified, please re-enter\n'
    sys.exit("Program exit")

#Create a pipeline that can directly feed the result from TF-IDF to SGD classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())])

#Create a list of parameters for vectorization, TF-IDF and SGD classifier
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__n_iter': (10, 50),
}

#Use grid search to find the best parameters
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(ingredients, trainLabel)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#Use the best parameters to predict
result = grid_search.predict(ingredientMatrix)

#write the result to file
write_to_file(outFile, testIDs, result)
