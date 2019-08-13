# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:04:43 2019

credits to Samira Munir https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386
Follow her github https://github.com/samiramunir/Simple-Sentiment-Analysis-using-NLTK

@author: Anirudh
"""

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import os,sys
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

train_path = 'imdb-movie-reviews-dataset/aclImdb/train/pos/'
train_path2 = 'imdb-movie-reviews-dataset/aclImdb/train/neg/'

files_pos = os.listdir(train_path)
files_pos = [open(train_path+f, 'r', encoding="utf8").read() for f in files_pos]

files_neg = os.listdir(train_path2)
files_neg = [open(train_path2+f, encoding="utf8").read() for f in files_neg]

len(files_neg) # Gives the number of the negative words in the trainng set 
len(files_pos) # Gives the number of the positive words in the trainng set 

all_words = []
documents = []

stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

############################### PRE PROCESSING ###############################
i =0
j= 0

for p in  files_pos:
    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())    
    print(i)
    i+= 1
    
    
for p in files_neg:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "neg") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speecwh tagging for each word 
    neg = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
    print(j)
    j+= 1

len(all_words) # gives the total word count 
len(documents)


##################### Creating a word cloud #######################################################
pos_A = []
for w in pos:
    if w[1][0] in allowed_word_types:
        pos_A.append(w[0].lower())
pos_N = []
for w in neg:
    if w[1][0] in allowed_word_types:
        pos_N.append(w[0].lower())
        
from wordcloud import WordCloud
text = ' '.join(pos_A)
wordcloud = WordCloud().generate(text)

plt.figure(figsize = (15, 9))
# Display the generated image:
plt.imshow(wordcloud, interpolation= "bilinear")
plt.axis("off")
plt.show()


###############################################################################
#########################  Finding the features ###############################
###############################################################################



# pickling the list documents to save future recalculations 
save_documents = open("Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

# creating a frequency distribution of each adjectives. 
BOW = nltk.FreqDist(all_words)
BOW

# listing the 5000 most frequent words
word_features = list(BOW.keys())[:5000]
word_features[0], word_features[-1]

len(word_features)

#  Pickling the features for later use
save_word_features = open("Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


#creating a frequency distribution of all words
all_words = nltk.FreqDist(all_words)

# listing the 5000 most frequent words
word_features = list(all_words.keys())[:5000]



# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features 
# The values of each key are either true or false for wether that feature appears in the review or not

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Creating features for each review

i= 0
featuresets = []
for (rev, category) in documents:
    featuresets.append((find_features(rev), category) )
    print(i)
    i += 1


#pickling the dataset 
path_feature = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/feature_sets.pickle'
save_feature_sets = open(path_feature,"wb")
pickle.dump(featuresets, save_feature_sets)
save_feature_sets.close()


training_set = featuresets[:20000]
testing_set = featuresets[20000:]
print( 'training_set :', len(training_set), 'testing_set :', len(testing_set))


############################################################################

# Naive Bayes Classifier  model 

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

mif = classifier.most_informative_features()

mif = [a for a,b in mif]
print(mif)

# getting predictions for the testing set by looping over each reviews featureset tuple
# The first elemnt of the tuple is the feature set and the second element is the label 
ground_truth = [r[1] for r in testing_set]

preds = [classifier.classify(r[0]) for r in testing_set]
from sklearn.metrics import f1_score
f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')


## Other models 
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set))*100)


### Pickling all models
def create_pickle(c, file_name): 
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

classifiers_dict = {'ONB': [classifier, 'pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'pickled_algos/SGD_clf.pickle'], 
                    'SVC': [SVC_clf, 'pickled_algos/SVC_clf.pickle']}




for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])