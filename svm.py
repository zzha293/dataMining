
# encoding: utf-8
"Single classifier using Linear SVC"

import sys
import json
import re
import csv
import numpy
import scipy
from sklearn import svm
from sklearn.svm import LinearSVC

# @param: fname<str>
# @return: ingredients<list<str>>
# @return: trainLabel<list<str>>
def dataReader(fname):

    "Function that read from a file and create list with all ingredients and a list of string with cuisine"

    with open(fname) as dataFile:
        data = json.load(dataFile)

    ingredients = set()
    trainLabel = []
    for i in range(0, len(data)):
        trainLabel.append(data[i]['cuisine'])
        for j in range(0, len(data[i]['ingredients'])):
            ingredientTemp = data[i]['ingredients'][j]
            ingredientTemp = removeSpecialCharacter(ingredientTemp)
            ingredientStr = ingredientTemp.encode("ascii","replace")
            ingredients.add(ingredientStr)

    ingredients = list(ingredients)

    return ingredients, trainLabel

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
    allIngredients, trainLabel = dataReader(train)
    matrix, trainIDs = createMatrix(train, allIngredients)
else:
    print 'No training data, please re-enter\n'
    sys.exit('Program exit')

if test is not None:
    ingredientMatrix, testIDs = createMatrix(test, allIngredients)
else:
    print 'No test data, please re-enter\n'
    sys.exit('Program exit')

if output is not None:
    # write into output file
    outFile = output
else:
    print 'No output file specified, please re-enter\n'
    sys.exit("Program exit")

#Train the linear classifiers
svm = LinearSVC(random_state=0)
svm.fit(matrix, trainLabel)
result = svm.predict(ingredientMatrix)

#Write the output to file
write_to_file(outFile, testIDs, result)



