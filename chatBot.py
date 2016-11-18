# Basic Program Needs
import collections
import operator
from collections import Counter
from copy import deepcopy
import math

#GlobalList
contentList =[]
countWords = collections.OrderedDict()

#take the file name and store all contents as a dictionary with input as Key and output as value
def readContentsFromFile(fileName):
    docDict =collections.OrderedDict()
    with open(fileName, "r") as files:
        for line in files:
            if(line != '\n'):
                getindividualContent = line.strip().split(":")
                docDict[getindividualContent[0]] =  getindividualContent[1]
    return docDict

#Take input as an String and return the unique word list with occurrence
def getWordsCount(inputValue):
    countDict = collections.OrderedDict()
    inputList = inputValue.strip().split(' ')
    countDict = Counter(inputList)
    return countDict

#Make Universal Matrix on the basis of documents
def makeUniversalMatrix(inputList,documentDict ):
    matrixList = []
    for key,value in documentDict.items():
        individualDict = getWordsCount(key)
        tempList=[]
        for keyI  in inputList:
            if keyI in individualDict:
                tempList.append(individualDict[keyI])
                if(keyI not in countWords):
                    countWords[keyI] = 1
                else:
                    countWords[keyI] = countWords[keyI]+1
            else:
                tempList.append(0)
        matrixList.append(deepcopy(tempList))
    return matrixList

def calculateTF(matrixList):
    tfmatrix = []
    for item in matrixList:
            individualList =[]
            for each in item:
                curr = float(each)
                if(each > 0):
                    new = 1 + math.log(each)
                else:
                    new = 0
                individualList.append(new)
            tfmatrix.append(deepcopy(individualList))
    return tfmatrix

def calculateIDF(tfmatrix, idflist):
    lastList = tfmatrix[-1]
    del tfmatrix[-1]
    tempList=[]
    for i in range(0,len(lastList)):
        tempList.append( float(idflist[i]) * float(lastList[i]) )
    tfmatrix.append(tempList)
    return tfmatrix

#Generate the matrix after Length Normalization
def calculateUnitVector(idfmatrix):
    unitMatrix =[]
    for item in idfmatrix:
            tempList=[]
            currentUnitVector = getUnitVector(item)
            for innerItem in item:
                if( currentUnitVector == 0):
                    tempList.append(float(innerItem))
                else:
                    tempList.append(float(innerItem) / currentUnitVector)
            unitMatrix.append(deepcopy(tempList))
    return unitMatrix

def calculateSimilarity(matrix):
    resultMatrix = collections.OrderedDict()
    query = matrix[-1]
    for i in range (0,len(matrix)-1):
        sum = 0
        curr = matrix[i]
        for j in range (0, len(query)):
            sum = sum +  ( float(query[j]) * float(curr[j])  )
        resultMatrix[i] = sum
    return resultMatrix

#Calculate the Unit matrix for the given List
def getUnitVector(item):
    sum = 0
    for i in range (0, len(item)):
        sum = sum + math.pow(float(item[i]), 2)
    return math.sqrt(sum)

#If word is not present in any document then give it highest value equal to number of document
def normalizeCountDictionary(inputList, countWords):
    for item in inputList:
        if item not in countWords:
            countWords[item] = totalDocument
    return countWords


#main Program starts from here
documentDict = readContentsFromFile("Greetings.txt")

#Parse the inout Query String
inputString = "hello"
inputStringDict = getWordsCount(inputString)
inputList = inputStringDict.keys()

#Claculate Term frequency for each document
matrixList = makeUniversalMatrix(inputList, documentDict)

#Append query as Document at the last matrix
queryList=[]
for key in inputList:
    queryList.append(inputStringDict[key])
matrixList.append(deepcopy(queryList))

#calculate tfscores for each matrix
tfmatrix = calculateTF(matrixList)

#Calculate the idf's
totalDocument = len(matrixList) #Including the query string also
idflist =[]
#print(countWords)
countWords = normalizeCountDictionary(inputList,countWords)

for key,value in countWords.items():
    idflist.append(math.log(  float(totalDocument) / float(value)  ))

idfmatrix = calculateIDF(deepcopy(tfmatrix), idflist)

finalScores = calculateUnitVector(deepcopy(idfmatrix ))
resultantMatrix = calculateSimilarity(deepcopy(finalScores ))

#get the top k values and output the result from the original List
matched =  list ( dict(sorted(resultantMatrix.items(), key=operator.itemgetter(1), reverse=True)[:1]).keys())
docList = list(documentDict.values())
outputString = docList[matched[0]]
print(outputString)

'''
print(documentDict)
print(matrixList)
print(tfmatrix)
print (idfmatrix)
print(finalScores)
print(resultantMatrix)
'''


