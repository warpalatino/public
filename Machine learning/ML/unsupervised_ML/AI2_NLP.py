import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re   # regular expressions
import nltk # natural language toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# sk packages - for bag of words and other stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# -------------------------
# REINFORCEMENT LEARNING
# Natural language processing



# load data
# ------------
dataset = pd.read_csv('../data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# cleaning text
# ------------
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)


# create bag of words model
# ------------
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# split dataset
# ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# run the model - via Naive Bayes
# ------------
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# predicting results vs test
# ------------
y_pred = classifier.predict(X_test)
results = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
# print(results)


# observing accuracy
# ------------
cm = confusion_matrix(y_test, y_pred)
# print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
