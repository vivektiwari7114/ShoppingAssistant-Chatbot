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


readFile('men_shirts.txt', 'men_shirts'); # 'r', encoding="latin1");
readFile('greetings.txt', 'greetings');
readFile('bye.txt','bye');

classifierNaive=nltk.classify.NaiveBayesClassifier.train(train);

fileContents=open('test.txt','r',encoding="latin1");
test=[];
for lines in fileContents:
    word=lines.split(":");
    currentSent=word[0];
    currentSent=word_tokenize(currentSent);
    test.append(Counter(currentSent));
fileContents.close();

#classifier=nltk.classify.NaiveBayesClassifier.train(train);
sorted(classifierNaive.labels());
print (classifierNaive.labels());
xx=classifierNaive.classify_many(test);
# read the file and add split on the : and store each lhs as param to dictionary q
print (xx);
