import os
import json
import warnings
import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer  
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split  
import demoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools

# nltk.download('wordnet')
# nltk.download('stopwords')
# demoji.download_codes()
data = pd.read_csv('./Final_represent/tweet_data_final.txt')

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')

# remove punctuation such as !, ?, #, $
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# remove stopwords
def remove_stopwords(text):
    # stop_words = set(stopwords.words('english'))
    f = open(r"./Final_represent/stopwords-en.txt", "r", encoding="utf-8")
    stop_words = f.read()
    words = [w for w in text if w not in stop_words]
    return words

# lemmatization
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

# stemming
def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

# make emoji to words
def emojitotext(text):
    emoji = demoji.findall(text)
    for emojicode, word in emoji.items():
        text = text.replace(emojicode, " "+word)
    return text

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, fontsize=6)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


data['text_parse'] = data['text']
data['text_parse'] = data['text_parse'].apply(lambda x: emojitotext(x))
data['text_parse'] = data['text_parse'].apply(lambda x: remove_punctuation(x))
data['text_parse'] = data['text_parse'].apply(lambda x: tokenizer.tokenize(x.lower()))
data['text_parse'] = data['text_parse'].apply(lambda x: remove_stopwords(x))
data['text_parse'] = data['text_parse'].apply(lambda x: word_lemmatizer(x))
data['text_parse'] = data['text_parse'].apply(lambda x: word_stemmer(x))

category_codes = {
    'Politics': 0,
    'Education': 1,
    'Health': 2,
    'Marketing': 3,
    'Music': 4,
    'News': 5,
    'Sport': 6,
    'Technology': 7,
    'Pets': 8,
    'Food': 9,
    'Family': 10
}

data['label_code'] = data['class']
data = data.replace({'label_code': category_codes})

X_train, X_test, y_train, y_test = train_test_split(
    data[['text', 'text_parse']], data['label_code'], test_size=0.20, random_state=42)

tfidf_vectorizer = TfidfVectorizer(
    min_df=1,
    max_features=720,
    strip_accents='unicode',
    analyzer='word',  
    ngram_range=(1, 1),
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    lowercase=True)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text_parse'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['text_parse'])

word_freq_data = pd.DataFrame(
    X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names())
top_words_data = pd.DataFrame(
    word_freq_data.sum()).sort_values(0, ascending=False)
# print(word_freq_data)
# print(top_words_data)
# file2 = open(r"temp1.txt","w+",encoding="utf-8")
# for word in top_words_data.index:
#     file2.writelines(word+'\n')
# file2.close()
# print (len(top_words_data))

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)
predictions = naive_bayes.predict(X_test_tfidf)

# print(classification_report(y_test, predictions,
#                             target_names=category_codes.keys()))

cm = confusion_matrix(y_test, predictions)
peraccuracy = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

peraccuracy = (TP+TN)/(TP+FP+FN+TN)
# print(peraccuracy)
perprecision = (TP)/(TP+FP)
perrecall = (TP)/(TP+FN)
perf1 = 2*perprecision*perrecall/(perprecision+perrecall)

columns = ['accuracy', 'precision', 'recall', 'F1']
finalmetrics = pd.DataFrame(index=category_codes.keys(), columns=columns)
finalmetrics = finalmetrics.fillna(0)
finalmetrics['accuracy'] = peraccuracy
finalmetrics['precision'] = perprecision
finalmetrics['recall'] = perrecall
finalmetrics['F1'] = perf1
finalmetrics = finalmetrics.round(2)
print(finalmetrics)

plot_confusion_matrix(cm, classes=category_codes.keys(), normalize=True,
                      title='confusion matrix')
