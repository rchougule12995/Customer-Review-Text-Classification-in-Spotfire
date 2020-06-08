import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string

dataset = data_table
# Function to count the percentage of puntuations in the text, before cleaning
def count_punct(review_text):
    count = sum([1 for char in review_text if char in string.punctuation])
    return round(count/(len(review_text) - review_text.count(" ")), 3)*100


# Function to get the length of the review text, and call punctuations percentage
# the length of text before cleaning is considered. (Actual review)
def review_length_punct(dataset):
	dataset["review_len"] = dataset["review"].apply(lambda x: len(x) - x.count(" "))
	dataset["punctuations"] = dataset["review"].apply(lambda x: count_punct(x))

	
review_length_punct(dataset)