#!/usr/bin/env python

'''
A Spam filter based on Naive Bayes Alg.
'''

import os
import io
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from joblib import Parallel, delayed
import random
import sys


def read_files(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fname in filenames:
            if(fname.find('cmds')!=-1):
                continue
            full_fname = os.path.join(dirpath, fname)
            fin = io.open(full_fname, encoding = 'latin-1')
            text_start = False
            lines = []
            for line in fin:
                if(text_start):
                    lines.append(line)
                else:
                    if(line=='\n'):
                        text_start = True
            yield full_fname, '\n'.join(lines)

def extract_data(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'label': classification})
        index.append(file_name)
    data_frame = DataFrame(rows, index=index)
    return data_frame

def train_fun(train_indices, test_indices, data, pipeline, confusion_matrix, f1_score, SPAM, HAM):
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['label'].values
    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['label'].values
    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)
    confusion = np.array([[0, 0], [0, 0]])
    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    return (confusion, score)

def download_corpus(url):
    import urllib
    print 'downloading and extracting \n'+url
    fname = url.split('/')[-1]
    urllib.urlretrieve (url, fname)
    import tarfile
    tar = tarfile.open(fname, mode='r:bz2')
    tar.extractall()
    tar.close()
    return
        
def main():
    
    np.random.seed(1236)
    url = "https://spamassassin.apache.org/publiccorpus/20030228_spam_2.tar.bz2"
    download_corpus(url)
    url = "https://spamassassin.apache.org/publiccorpus/20030228_easy_ham.tar.bz2"
    download_corpus(url)

    print
    print 'Starting the Spam filter code:'
    data_files = [['spam_2',        'spam'],
                  ['easy_ham',    'ham']]
    
    data = DataFrame({'text': [], 'label': []})
    for path, classification in data_files:
        data = data.append(extract_data(path, classification))
    data = data.reindex(np.random.permutation(data.index))


    pipeline = Pipeline([
        ('vectorizer',  HashingVectorizer(non_negative=True, ngram_range=(1, 2))),
        ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',  MultinomialNB()) ])

    
    k_fold = KFold(n=len(data), n_folds=6)    
    (results) = Parallel(n_jobs=4)(delayed(train_fun)(train_indices, test_indices, data, pipeline, confusion_matrix, f1_score, 'spam', 'ham') for (train_indices, test_indices) in k_fold)
    

    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    for confusion_item, scores_item in results:
        confusion = confusion+confusion_item
        scores.append(scores_item)
    print('Total emails classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
    
if __name__ == "__main__":
    main()
        
