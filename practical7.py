#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[9]:


get_ipython().system('python3 -m spacy download en_core_web_sm')


# In[10]:


import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Process a text
doc = nlp("This is a simple NLP test using SpaCy.")

# Print out tokens
print("Tokens:")
for token in doc:
    print(token.text)


# In[11]:


documents = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Apple’s new product might revolutionize the smartphone industry.",
    "Startup founders in California are optimistic about funding opportunities."
]


# In[12]:


import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

preprocessed_docs = []

for doc in documents:
    spacy_doc = nlp(doc)
    tokens_info = []

    for token in spacy_doc:
        if token.text.lower() not in stop_words and token.text not in string.punctuation:
            info = {
                "token": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "stem": stemmer.stem(token.text)
            }
            tokens_info.append(info)
    
    preprocessed_docs.append(tokens_info)

# Show processed output for first doc
print("--- Preprocessing Output ---")
for token in preprocessed_docs[0]:
    print(token)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Simple preprocessing: remove stopwords and punctuation
def simple_clean(doc):
    doc_nlp = nlp(doc)
    return " ".join([token.lemma_ for token in doc_nlp if token.text.lower() not in stop_words and token.text not in string.punctuation])

cleaned_documents = [simple_clean(doc) for doc in documents]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_documents)

# Display TF-IDF matrix
print("\n--- TF-IDF Matrix ---")
import pandas as pd

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df)


# In[ ]:




