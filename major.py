"Majority vote of all the results"
import csv
import math
from sklearn import preprocessing
from sklearn import linear_model

allcuisine = {}
result1 = {}
result2 = {}
result3 = {}
result4 = {}
result5 = {}
result6 = {}
result7 = {}
result8 = {}
result9 = {}
result10 = {}

with open('svmOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result1[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('sgdOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result2[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('randFOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result3[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('bayesOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result4[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('nnOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result5[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value


with open('logReOutput.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result6[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value



with open('predict_4nn.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result7[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('predict_4reg.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result8[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value


with open('predict_4svm.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result9[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value

with open('predict_4sgd.csv', 'r') as csvfile:
    next(csvfile)
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        key = row[0]
        value = row[1]
        result10[key] = value
        if value not in allcuisine.values():
            allcuisine[len(allcuisine) + 1] = value


writeFile = open('majorOutput.csv', 'w')
writeFile.write("id,cuisine\n")

tempResult = []
tempResult.append(result1)
tempResult.append(result2)
tempResult.append(result3)
tempResult.append(result4)
tempResult.append(result5)
tempResult.append(result6)
tempResult.append(result7)
tempResult.append(result8)
tempResult.append(result9)
tempResult.append(result10)


for key in result1.keys():
    countDict = {}
    for i in range(0, len(tempResult)):
        if tempResult[i][key] not in countDict:
            countDict[tempResult[i][key]] = 1
        else:
            countDict[tempResult[i][key]] += 1


    #print countDict
    max = 0
    finalCuisine = ''
    for key1, value1 in countDict.iteritems():
        if value1 > max:
            max = value1
            finalCuisine = key1

    #print finalCuisine
    writeFile.write("%s,%s\n" % (key, finalCuisine))

writeFile.close()

