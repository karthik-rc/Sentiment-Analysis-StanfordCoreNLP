# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:58:13 2019

@author: rckar
"""
"""-------------------------------------------------------------------------------
This is a demo of a basic off the shelf model of StanfordNLP's sentiment analyzer
----------------------------------------------------------------------------------
"""

# Establish the connection to StanfordCoreNLP server through command prompt

# import StanfordCoreNLP and initiate the connection
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://127.0.0.1:9000')


def analyze_sentiment(sent):

    # Run the sentiment analyzer model that also tokenizes, splits, parses and lemmatizes sentences
    output = nlp.annotate(sent, properties={
      'annotators': 'tokenize,ssplit,parse,lemma,sentiment',
      'outputFormat': 'json'
      })
    
    # Outputs the sentiment and the related sentiment value on a number scale
    return output['sentences'][0]['sentiment'], output['sentences'][0]['sentimentValue']


# Input sentences from users to analyze sentences
sent = input("Enter your sentence: ")

# Call the function
analyze_sentiment(sent)



