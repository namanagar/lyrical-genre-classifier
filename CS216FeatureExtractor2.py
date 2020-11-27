## TERMINAL 2
import json
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

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

all_songs_df = pd.DataFrame(allSongsInCorpus)
all_songs_df.columns = ['Index', 'Song', 'Year', 'Artist', 'genre', 'lyrics', 'lyricsperiod', 'lyricsStemmed']
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for genre in ["Hip-Hop","Metal","Pop","Rock","Country"]:
    subset = all_songs_df[
        (all_songs_df.genre == genre)]
    train_set = subset.sample(n=700, random_state=200)
    test_set = subset.drop(train_set.index)
    
    train_df = train_df.append(train_set)
    test_df = test_df.append(test_set)

train_df = shuffle(train_df)
test_df = shuffle(test_df)

text_clf1 = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', MultinomialNB(alpha=0.1))])
"""
text_clf2 = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', SVC(kernel="linear", C=0.025))])
"""


text_clf1.fit(train_df.lyrics, train_df.genre)
#text_clf2.fit(train_df.lyrics, train_df.genre)

predicted1 = text_clf1.predict(test_df.lyrics)
cm = confusion_matrix(test_df.genre, predicted1)

fig,ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=["Hip-Hop","Metal","Pop","Rock","Country"], yticklabels=["Hip-Hop","Metal","Pop","Rock","Country"],
           title="Confusion Matrix of Multinomial Bayes Classifier",
           ylabel='True label',
           xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.show()
#predicted2 = text_clf2.predict(test_df.lyrics)

#text_clf1.fit(train_df.lyricsStemmed, train_df.genre)
#text_clf2.fit(train_df.lyricsStemmed, train_df.genre)

#predicted4 = text_clf1.predict(test_df.lyricsStemmed)
#predicted5 = text_clf2.predict(test_df.lyricsStemmed)

accuracy1 = np.mean(predicted1 == test_df.genre)
#accuracy2 = np.mean(predicted2 == test_df.genre)

#accuracy4 = np.mean(predicted4 == test_df.genre)
#accuracy5 = np.mean(predicted5 == test_df.genre)

#print("The first lemmatized is: " + str(accuracy1))
#print("The second lemmatized is: " + str(accuracy2))

#print("The first stemmed is: " + str(accuracy4))
#print("The second stemmed is: " + str(accuracy5))

#predicted_list1 = [accuracy1, accuracy2]
#predicted_list2 = [accuracy4, accuracy5]
"""
fig,ax = plt.subplots()
ind = np.arange(2)
width = 0.40
p1 = ax.bar(ind, predicted_list1, width, color='r')
p2 = ax.bar(ind+width, predicted_list2, width, color='b')
ax.set_title('Performance across different models')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Multinomial NB','Random Forest'))
ax.legend((p1[0], p2[0]),('Lemmatized', 'Stemmed'))
ax.yaxis.set_units("Accuracy")
ax.autoscale_view()
ax.axhline(y=0.25, color='g')

plt.show()

label =['CountVectorizer','TFIDF', 'Random Guessing']
index = np.arange(3)
plt.bar(index, predicted_list)
plt.xlabel('Features Used', fontsize=7)
plt.ylabel('Accuracy', fontsize =7)
plt.xticks(index, label, fontsize=7, rotation=30)
plt.title('Accuracy for different kinds of feature vectors')
plt.show()
"""




