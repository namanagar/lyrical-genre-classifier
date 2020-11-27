from IPython import display
import math
from pprint import pprint
import os
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import lyricsgenius
from lyricsgenius.artist import Artist
import csv
import io
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

sns.set(style='darkgrid', context='talk', palette='Dark2')
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

with open('uncleaned_lyrics.csv', 'r', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    line_count= 0
    count = 0
    cleaned_strings = list()
    
    for row in csv_reader:
        if line_count == 0:
            line_count +=1
        else:
            index = str(row[0])
            song = str(row[1])
            year = str(row[2])
            artist = str(row[3])
            genre = str(row[4])
            lyrics = str(row[5])
            if(len(year) < 2 or len(artist) < 2 or len(genre) < 2 or len(lyrics) < 2):
                continue
            elif (genre == "Not Available" or genre == "Other" or genre == "Jazz" or genre =="Electronic" or genre=="Folk" or genre=="R&B" or genre=="Indie"):
                continue
            else:
                count = count + 1
                
                # Replace weird characters and brackets
                lyrics = lyrics.replace("\n", " ")
                lyrics = lyrics.replace(",", "")
                lyrics = lyrics.replace(")", "")
                lyrics = lyrics.replace("(", "")
                lyrics = lyrics.replace("[", "")
                lyrics = lyrics.replace("]", "")
                lyricsWithPeriod = lyrics
                lyrics = lyrics.replace(".", "")
                lyrics = lyrics.replace("?", "")
                lyrics = lyrics.replace("!", "")
                lyrics = lyrics.replace("'", "")
                lyrics = lyrics.replace("\"", "\"")
                
                lyrics = re.sub(r"\[.*?\]", "", lyrics)
                lyrics = re.sub(r"\{.*?\}", "", lyrics)

                # Tokenize and remove stopwords
                word_tokens = lyrics.split(" ")
                lyrics = [w for w in word_tokens if not w in stop_words]

                # Tokenize and lemmatize words
                lyrics_lem = [wordnet_lemmatizer.lemmatize(w) for w in lyrics]

                # Tokenize and stem words
                lyrics_stem = [ps.stem(w) for w in lyrics]

                # To lowercase
                lyrics_lem = [w.lower() for w in lyrics_lem]
                lyrics_stem = [w.lower() for w in lyrics_stem]

                # Joins the lyrics back into one string
                separator = " "
                lyrics_lem = separator.join(lyrics_lem)
                lyrics_stem = separator.join(lyrics_stem)
                    
                final_arr = [index, song, year, artist, genre, lyrics_lem, lyricsWithPeriod, lyrics_stem]
                cleaned_strings.append(final_arr)


print('there are: ' + str(count) + ' valid rows')
count2 = 0
myDict = {}
with io.open('cleaned_lyrics.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for line in cleaned_strings:
        if(line[4] in myDict):
           myDict[line[4]] = myDict[line[4]]+1
        else:
           myDict[line[4]] = 0
           
        writer.writerow(line)
        count2+=1

for key,val in myDict.items():
    keyToUse = str(key)
    valToUse = str(val)
    print(keyToUse + " has " + valToUse)


    

