"""
Question
Given set of reviews identify which review is positive and which is negative
using bag of words NLP algorithm
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# we are using tsv insteal of csv because Main difference between csv and tsv is 
# csv has comma that seperates the columns
# but in our data we have reviews of people of which might contains comma
#hence python will not be able to find colum properly hence we are using tsv
#which uses tab instead of comma to differentaiate colum and tab are not used while
# writing reviews hence it is great

# NLP Preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# we are calling list of review corpus because corpus means group of similar things
corpus=[]
for i in range(0,1000):
    # remove all character other a-z and A-Z
    review=re.sub('[^a-zA-z]'," ",dataset['Review'][i])
    #convert to lower case
    review=review.lower()
    #splitting into list of words
    review=review.split()
    # removing stop words and stemming
    stemmed=[]
    # i dont think not should be removed because it important for deciding
    #feedback is positive or negative
    for word in review:
        if word != 'not':
            if not word in set(stopwords.words('english')):
                stemmed.append(ps.stem(word))
        else:
            stemmed.append(word)
            
    review=stemmed
    #Joining words to form another string
    review=' '.join(review)
    corpus.append(review)

#Create SParse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""
FP=FAlse Positive TP=True Positive FN= False negative TN= True NEgative
Comparison
Bayes FP=42 FN=12 TP=55 TN=91 F1 SCore=
Decision Tree FP=19 FN=31 TP=78 TN=72
Random FOrest FP=7 FN=38 TP=90 TN=65

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

Best F1 Score decide which to use

"""
''''
# Random FOrest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=700,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)'''

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)