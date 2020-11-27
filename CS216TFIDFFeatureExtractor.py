## TERMINAL 2
import json
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

allWordsInCorpusSet= set()
allWordsInCorpus = list()

countHipHop = 0
countMetal = 0
countPop = 0
countRock = 0
countCountry = 0

allSongsInCorpus = list()

with open('hip_hop_sample.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    for row in csv_reader:
        lyrics = str(row[5])
        words = lyrics.split(" ")
        countHipHop += 1
        allSongsInCorpus.append(row)
        for word in words:
            if(word not in allWordsInCorpusSet):
                allWordsInCorpusSet.add(word)
                allWordsInCorpus.append(word)

print(1)

with open('metal_sample.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    for row in csv_reader:
        lyrics = str(row[5])
        words = lyrics.split(" ")
        countMetal += 1
        allSongsInCorpus.append(row)
        for word in words:
            if(word not in allWordsInCorpusSet):
                allWordsInCorpusSet.add(word)
                allWordsInCorpus.append(word)

print(2)

with open('pop_sample.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    for row in csv_reader:
        lyrics = str(row[5])
        words = lyrics.split(" ")
        countPop += 1
        allSongsInCorpus.append(row)
        for word in words:
            if(word not in allWordsInCorpusSet):
                allWordsInCorpusSet.add(word)
                allWordsInCorpus.append(word)

print(3)

with open('rock_sample.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    for row in csv_reader:
        lyrics = str(row[5])
        words = lyrics.split(" ")
        countRock +=1
        allSongsInCorpus.append(row)
        for word in words:
            if(word not in allWordsInCorpusSet):
                allWordsInCorpusSet.add(word)
                allWordsInCorpus.append(word)

print(4)

with open('country_sample.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    for row in csv_reader:
        lyrics = str(row[5])
        words = lyrics.split(" ")
        countCountry+=1
        allSongsInCorpus.append(row)
        for word in words:
            if(word not in allWordsInCorpusSet):
                allWordsInCorpusSet.add(word)
                allWordsInCorpus.append(word)

totalNumDocuments = countHipHop+ countMetal + countPop + countRock + countCountry
print(totalNumDocuments == len(allSongsInCorpus))
print(len(allWordsInCorpus))
print(len(allWordsInCorpus) == len(allWordsInCorpusSet))

# Create 2 parallel arrays: one of ground truth another of tuple representing feature vector
# Each song has a tuple representing its feature vector
ground_truth_vector = list()
feature_list = list()

idf_dict = {}
le = preprocessing.LabelEncoder()

print('creating IDF stuff')
for song in allSongsInCorpus:
    lyrics = song[5]
    individual_words = set(lyrics.split(" "))

    for word in individual_words:
        if(word in allWordsInCorpusSet and word not in idf_dict):
            idf_dict[word] = 1
        elif(word in allWordsInCorpusSet and word in idf_dict):
            idf_dict[word] = idf_dict[word]+1
    """
    for word in allWordsInCorpus:
        term_present 
        
        if(term_freq > 0 and word not in idf_dict):
            idf_dict[word] = 1
        elif(term_freq > 0 and word in idf_dict):
            idf_dict[word] = idf_dict[word] + 1
    """

print('creating TF stuff')
count = 0
for song in allSongsInCorpus:
    tf_idf_vector = list()
    lyrics = song[5]
    individual_words = lyrics.split(" ")
    individual_words_set = set(individual_words)
    term_freq = 0


    for word in allWordsInCorpus:
        if(word not in individual_words_set):
            tf_idf_vector.append(0)
        else:
            term_freq = individual_words.count(word)
            tf_value = term_freq/len(individual_words)
            tf_idf_vector.append(tf_value/np.log(totalNumDocuments / idf_dict[word]))
        
    feature_list.append(tuple(tf_idf_vector))
    ground_truth_vector.append(song[4])
    count = count + 1

ground_truth_encoded = le.fit_transform(ground_truth_vector)
print(ground_truth_encoded[200])

print('building model')
X_train, X_test, y_train, y_test = train_test_split(feature_list, ground_truth_encoded, test_size=0.3,random_state=109)
gnb = GaussianNB()
print('training model')
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))




            


