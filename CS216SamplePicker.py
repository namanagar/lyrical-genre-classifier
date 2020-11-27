import json
import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics
from random import shuffle
import io


allHipHop = list()
allRock = list()
allCountry = list()
allMetal = list()
allPop = list()


with open('cleaned_lyrics.csv', 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')

    for row in csv_reader:
        if(str(row[4]) == "Hip-Hop"):
            allHipHop.append(row)
        elif(str(row[4]) == "Rock"):
            allRock.append(row)
        elif(str(row[4]) == "Country"):
            allCountry.append(row)
        elif(str(row[4]) == "Metal"):
            allMetal.append(row)
        elif(str(row[4]) == "Pop"):
            allPop.append(row)

shuffle(allHipHop)
shuffle(allRock)
shuffle(allCountry)
shuffle(allMetal)
shuffle(allPop)

sampleHipHop = allHipHop[:10000]
sampleRock = allRock[:10000]
sampleCountry = allCountry[:10000]
sampleMetal = allMetal[:10000]
samplePop = allPop[:10000]

print(len(sampleHipHop))

with io.open('hip_hop_sample.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for entry in sampleHipHop:
        writer.writerow(entry)

print('finished sampling hip hop')

with io.open('rock_sample.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for entry in sampleRock:
        writer.writerow(entry)

print('finished sampling rock')

with io.open('country_sample.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for entry in sampleCountry:
        writer.writerow(entry)

print('finished sampling country')

with io.open('metal_sample.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for entry in sampleMetal:
        writer.writerow(entry)

print('finished sampling metal')

with io.open('pop_sample.csv', 'w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    for entry in samplePop:
        writer.writerow(entry)

print('finished sampling pop')
      

