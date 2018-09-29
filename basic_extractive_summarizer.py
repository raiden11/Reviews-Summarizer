import pandas as pd
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from nltk.stem.porter import *
import string
import nltk

with open("/home/ayush/Documents/reviews-summarizer/reviews_light.txt") as file :

	all_reviews = file.readlines()

stop_words = stopwords.words('english')
stop_words += string.punctuation
stemmer = PorterStemmer()

for review in all_reviews:
    text = literal_eval(review)
    sents = sent_tokenize(text['reviewText'])
    word_sent = [word_tokenize(s.lower()) for s in sents]
    word_stop_remove = []
    print(word_sent)
    print(len(word_sent))
    for i in range(len(word_sent)):
        temp = []
        for j in range(len(word_sent[i])):
            if word_sent[i][j] not in stop_words:
                temp.append(word_sent[i][j])
        word_stop_remove.append(temp)

    word_stem = []
    for i in range(len(word_stop_remove)):
        temp = []
        for j in range(len(word_stop_remove[i])):
            #print(word_stop_remove[i][j])
            temp.append(stemmer.stem(word_stop_remove[i][j]))
            #print(stemmer.stem(word_stop_remove[i][j]))
        word_stem.append(temp)
    #print(word_stem)

    word_pos = []
    for i in range(len(word_stem)):
        word_pos.append(nltk.pos_tag(word_stem[i]))
    print(word_pos)

