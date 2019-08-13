#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:43:34 2019

@author: aadarsh
"""

"""-------------------------------------------------------------------------------
This is a demo of a basic off the shelf model of TextBlob's sentiment analyzer
----------------------------------------------------------------------------------
"""

# import TextBlob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

def analyze_sentiment_tb_default(sent):

    blob = TextBlob(sent)
    return blob.sentiment

def analyze_sentiment_tb_Naive(sent):

    blob = TextBlob(sent, analyzer = NaiveBayesAnalyzer())
    return blob.sentiment

# Input sentences from users to analyze sentences
sent = input("Enter your sentence: ")

# Call the function
analyze_sentiment_tb_default(sent)
analyze_sentiment_tb_Naive(sent)
