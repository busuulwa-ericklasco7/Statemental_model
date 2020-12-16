import pandas as pd
import numpy as np
import spacy
import gzip
import dill
from html import unescape
from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("./sentiment.csv", error_bad_lines = False)


tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

from html import unescape
def preprocessor(doc):
    return unescape(doc).lower()

nlp = spacy.load("en_core_web_sm")

nlp = spacy.load("en_core_web_sm", disable = ["ner", "paser", "tagger"])

def lemmatizer(doc):
    return [word.lemma_ for word in nlp(doc)]

STOP_WORDS_lemma = [word.lemma_ for word in nlp(" ".join(list(STOP_WORDS)))]
STOP_WORDS_lemma = set(STOP_WORDS_lemma).union({",", ".", ";"})

vectorizer = HashingVectorizer(preprocessor=preprocessor, alternate_sign=False, stop_words = STOP_WORDS
                               #tokenizer = lemmatizer, #ngram_range = (1,2),
                               )

clf = MultinomialNB()
pipe = Pipeline([("vectorizer", vectorizer), ("classifier", clf)])

X = df["SentimentText"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )

pipe.fit(X_train, y_train);
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)

#here am dumping the model into a file
with gzip.open("sentimental_mode.dill.gz", "wb") as f:
    dill.dump(pipe, f, recurse=True)
    
#here am reading the:fill to make predictions from the file with
#out training
with gzip.open("sentimental_mode.dill.gz", "rb") as f:
    model = dill.load(f)
    
#here am reading the:fill to make predictions from the file with
#out training
with gzip.open("sentimental_mode.dill.gz", "rb") as f:

model = dill.load(f);







