import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from lime.lime_text import LimeTextExplainer

data = pd.read_csv("IMDB.csv", index_col=0)

stwords = stopwords
special_char=[",",":"," ",";",".","?"]


vectorizer = CountVectorizer()
vectorizer.fit(data)
x = vectorizer.transform(data)

print(x)
print("Vocabulario (características):", vectorizer.get_feature_names_out())
print("Matriz BoW:", x.toarray())


