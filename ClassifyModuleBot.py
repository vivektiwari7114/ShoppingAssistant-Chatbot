import nltk;
import numpy;
from nltk.tokenize import word_tokenize;
from nltk.tokenize import wordpunct_tokenize;
from nltk.tokenize import sent_tokenize, word_tokenize;
from collections import *;

train=[];

# train = [(dict(a=1,b=1,c=1), 'y'),(dict(a=1,b=1,c=1), 'x'),(dict(a=1,b=1,c=0), 'y'),
# (dict(a=0,b=1,c=1), 'x'),(dict(a=0,b=1,c=1), 'y'),(dict(a=0,b=0,c=1), 'y'),
# (dict(a=0,b=1,c=0), 'x'),(dict(a=0,b=0,c=0), 'x'),(dict(a=0,b=1,c=1), 'y')];
test = [(dict(a=1,b=0,c=1)), # unseen
(dict(a=1,b=0,c=0)), # unseen
(dict(a=0,b=1,c=1)), # seen 3 times, labels=y,y,x
(dict(a=0,b=1,c=0))]; # seen 1 time, label=x]
#Test the Naive Bayes classifier:

def readFile(fileName, tag):
    global train;
    fileContents=open(fileName,'r', encoding="latin1");
    for lines in fileContents:
        word=lines.split(":");
        currentSent=word[0];
        currentSent=word_tokenize(currentSent);
        train.append((Counter(currentSent),tag));
    fileContents.close();

#Train data to categorize the input string

readFile('data/Greetings.txt', 'Greetings.txt');
readFile('data/Men_Shirts.txt', 'Men_Shirts.txt'); # 'r', encoding="latin1");
readFile('data/Women_Shirts.txt', 'Women_Shirts.txt');
readFile('data/Men_Jeans.txt','Men_Jeans.txt');
readFile('data/Women_Jeans.txt', 'Women_Jeans.txt'); # 'r', encoding="latin1");
readFile('data/Bye.txt','Bye.txt');

classifierNaive=nltk.classify.NaiveBayesClassifier.train(train);

fileContents=open('query.txt','r',encoding="latin1");
test=[];
queryData =[]
for lines in fileContents:
    #word=lines.split(" ");
    #currentSent=word[0];
    queryData.append(lines.strip() )
    currentSent=word_tokenize(lines);
    test.append(Counter(currentSent));
fileContents.close();

#classifier=nltk.classify.NaiveBayesClassifier.train(train);
sorted(classifierNaive.labels());
searchFile=classifierNaive.classify_many(test);
# read the file and add split on the : and store each lhs as param to dictionary q
#Name of the output file that consist of the name of the file where to search
classifierFile = "classifier.txt"
outputHandle = open(classifierFile, "w", encoding="latin1")
for input in queryData:
    outputHandle.write(input)
    outputHandle.write('\n')
for item in searchFile:
    outputHandle.write(item)
    outputHandle.write('\n')


