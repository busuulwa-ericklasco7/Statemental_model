#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy


# # Sentimental model
# 
# i have downloaded the data from this site #"http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip"

# In[2]:


#%%bash 
#wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip -nc -P /.


# In[3]:


df = pd.read_csv("./sentiment.csv", error_bad_lines = False)


# In[4]:


df.head()


# In[5]:


df['SentimentText'][:10]


# In[6]:


from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk import TweetTokenizer


# In[7]:


tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)


# In[8]:


tweet_tokenizer.tokenize(df.iloc[0, -1])


# In[9]:


df.iloc[0, -1]


# In[10]:


from html import unescape
def preprocessor(doc):
    return unescape(doc).lower()


# In[11]:


unescape("&lt")


# In[12]:


nlp = spacy.load("en_core_web_sm")


# In[13]:


nlp = spacy.load("en_core_web_sm", disable = ["ner", "paser", "tagger"])


# In[14]:


def lemmatizer(doc):
    return [word.lemma_ for word in nlp(doc)]


# In[15]:


from spacy.lang.en import STOP_WORDS


# In[16]:


STOP_WORDS_lemma = [word.lemma_ for word in nlp(" ".join(list(STOP_WORDS)))]
STOP_WORDS_lemma = set(STOP_WORDS_lemma).union({",", ".", ";"})


# In[17]:


from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[18]:


vectorizer = HashingVectorizer(preprocessor=preprocessor,
                               #tokenizer = lemmatizer,
                             alternate_sign=False,
                             
                            #ngram_range = (1,2),
                            stop_words = STOP_WORDS)

clf = MultinomialNB()
pipe = Pipeline([("vectorizer", vectorizer), ("classifier", clf)])


# In[19]:


X = df["SentimentText"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )


# In[ ]:


pipe.fit(X_train, y_train);


# In[ ]:


pipe.score(X_train, y_train)


# In[ ]:


pipe.score(X_test, y_test)


# In[20]:


import gzip
import dill


# In[22]:


#here am reading the:fill to make predictions from the file with
#out training
with gzip.open("sentimental_mode.dill.gz", "rb") as f:
    model = dill.load(f)

model.score(X_test, y_test);


# In[ ]:


#here am dumping the model into a file
with gzip.open("sentimental_mode.dill.gz", "wb") as f:
    dill.dump(pipe, f, recurse=True)


# In[23]:





# In[ ]:




