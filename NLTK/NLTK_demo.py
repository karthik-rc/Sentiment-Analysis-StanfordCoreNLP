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

# Loading Pickled document into memory 
pathr = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/documents.pickle'
documents_f = open(pathr, "rb")
documents = pickle.load(documents_f)
documents_f.close()

# Loading feature extractor into memory
featurer = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/word_features5k.pickle'
word_features5k_f = open(featurer, "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()



# Load all classifiers from the pickled files
    
# function to load models given filepath
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


path_onb = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/ONB_clf.pickle'
path_mnb = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/MNB_clf.pickle'
path_bnb = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/BNB_clf.pickle'
path_logreg = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/LogReg_clf.pickle'
path_sgd = 'Simple-Sentiment-Analysis-using-NLTK-master/pickled_algos/SGD_clf.pickle'


# Original Naive Bayes Classifier
ONB_Clf = load_model(path_onb)

# Multinomial Naive Bayes Classifier 
MNB_Clf = load_model(path_mnb)


# Bernoulli  Naive Bayes Classifier 
BNB_Clf = load_model(path_bnb)

# Logistic Regression Classifier 
LogReg_Clf = load_model(path_logreg)

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model(path_sgd)


# Class that helps choose the best model 

class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
def parse_text(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

### calling the class 
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)


# checks the sentiment of the text passed into the model 
def sentiment(text):
    feats = parse_text(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)




text_a = "Those who find ugly meanings in beautiful things are corrupt without being charming"
sentiment(text_a)

text_b = "The movie was meh"
sentiment(text_b)
     
