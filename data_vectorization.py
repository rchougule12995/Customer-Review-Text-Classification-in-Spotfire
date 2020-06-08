import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

dataset1 = dataset
original_ds = dataset.copy()
dataset1 = dataset1.drop(['review_len','punctuations'], axis=1)

# Function to lower the capitalized words, tokenizing words, removing stop words, 
# lemmatizing words in the review and returning the review
def clean_text_review(review_text):
    review_text = "".join([word.lower() for word in review_text if word not in string.punctuation and not word.isdigit()])
    review_text = review_text.strip()
    tokens = re.split('\W+', review_text)
    review = [word for word in tokens if word not in set(stopwords.words('english'))]
    wn = nltk.WordNetLemmatizer()
    review = [wn.lemmatize(word) for word in review]
    return review

# Function to normalize the review length and punctuations attribute in the dataset
def _normalize_length_punct():
    scaler = MinMaxScaler()
    temp = X_features['review_len']
    temp = temp.to_numpy()
    temp = temp.reshape(-1,1)
    X_features['review_len'] = scaler.fit_transform(temp)
    scaler = MinMaxScaler()
    temp = X_features['punctuations']
    temp = temp.to_numpy()
    temp = temp.reshape(-1,1)
    X_features['punctuations'] = scaler.fit_transform(temp)

# Function for tf-idf vectorization of the reviews after cleaning the data
def review_Vectorization(dataset):
    #setting max features to top 2350 as most of the datasets here do not similar set features after vectorization
    tfidf_vect = TfidfVectorizer(analyzer=clean_text_review, max_features=2350)
    # Setting the max_features after thorough testing on different models
    X_tfidf = tfidf_vect.fit_transform(dataset['review'])
    X_features = pd.concat([original_ds['review_len'], original_ds['punctuations'], pd.DataFrame(X_tfidf.toarray())], axis=1)
    return X_features
	
X_features = review_Vectorization(dataset1)
_normalize_length_punct()