import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

## Importing Textblob package
from textblob import TextBlob
from textblob import Word

# Importing CountVectorizer for sparse matrix/ngrams frequencies
from sklearn.feature_extraction.text import CountVectorizer

## Import datetime
import datetime as dt



import nltk.compat
import itertools
import chardet


##### Read the data file
#filepath= "C:\Users\asus\Desktop\DEGREE LECTURE NOTES\SEMESTER 4\Final Project\New folder (3)\test\src\issues.csv"
train_incidents = pd.read_csv(r"issues.csv",encoding="Windows-1252")


train_incidents["Subject"] = train_incidents["Subject"].apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
train_incidents["Subject"].head(5)

Short_description_most_freq_words = pd.Series(" ".join(train_incidents["Subject"]).split()).value_counts()
Short_description_most_freq_words.head(20)

sd_freq_plot = Short_description_most_freq_words.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Number Of Words")

plt.style.use("ggplot")
sd_freq_plot.set_xlabel("Frequency")
sd_freq_plot.set_ylabel("Terms")

totals = []
for i in sd_freq_plot.patches:
    totals.append(i.get_width())

for i in sd_freq_plot.patches:
    sd_freq_plot.text(i.get_width()+.3,i.get_y()+0.1,str(i.get_width()),fontsize = 8,color= 'black')

#print(sd_freq_plot)
#fig = sd_freq_plot.figure()
#ax = fig.add_subplot(111)
#ax.plot([1,2,3])
#fig.savefig('test.png')

bigrams = TextBlob(" ".join(train_incidents["Subject"])).ngrams(2)
word_vectorizer = CountVectorizer(ngram_range=(7,7), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(train_incidents["Subject"])
frequencies = sum(sparse_matrix).toarray()[0]
bi_grams_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

bi_grams_df.sort_values(by = "frequency",ascending=False).head(20)
    
plt.style.use("ggplot")
plt.xlabel("Frequency",)
plt.ylabel("Terms")
top20_bigrams = bi_grams_df["frequency"].sort_values(ascending = False).head(20)

top20_bigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Bi Grams")

plt.show()