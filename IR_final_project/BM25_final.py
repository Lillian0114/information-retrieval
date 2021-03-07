from sklearn.metrics.classification import accuracy_score
import os
import json
import warnings
import string
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords  # such as 'the', 'a', 'an', 'in'
from nltk.tokenize import RegexpTokenizer  # split words
from nltk.stem import WordNetLemmatizer  # convert words to root words
from nltk.stem.porter import PorterStemmer  # e.g. convert is, am, are -> be
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split  # split data to train and test
import demoji

import sklearn.metrics as skm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import joblib  # for save or load model
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from sklearn.feature_extraction.text import _document_frequency
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import itertools

class BM25Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, norm='l2', use_idf=False, use_bm25idf=False, smooth_idf=False,
                 delta_idf=False, sublinear_tf=True, bm25_tf=True, k=2.0, b=0.75
                 # ,sublinear_tf=False, use_idf=True, smooth_idf=True
                 ):
        self.norm = norm
        self.use_idf = use_idf
        self.use_bm25idf = use_bm25idf
        self.smooth_idf = smooth_idf
        # Required for delta idf's
        self.delta_idf = delta_idf

        self.sublinear_tf = sublinear_tf
        self.bm25_tf = bm25_tf
        self.k = k
        self.b = b

    def fit(self, X, y=None):

        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        if self.use_idf:
            n_samples, n_features = X.shape

            # BM25 idf
            if self.use_bm25idf:
                if self.delta_idf:
                    if y is None:
                        raise ValueError("Labels are needed to determine Delta idf")

                    N1, df1, N2, df2 = _class_frequencies(X, y)
                    delta_bm25idf = np.log(((N1 - df1 + 0.5) * df2 + 0.5) / ((N2 - df2 + 0.5) * df1 + 0.5))
                    self._idf_diag = sp.spdiags(delta_bm25idf,
                                                diags=0, m=n_features, n=n_features)
                else:
                    # vanilla bm25 idf
                    df = _document_frequency(X)

                    # perform idf smoothing if required
                    df += int(self.smooth_idf)
                    n_samples += int(self.smooth_idf)

                    # log1p instead of log makes sure terms with zero idf don't get
                    # suppressed entirely

                    bm25idf = np.log((n_samples - df) / df)


                    # bm25idf = np.log((n_samples - df + 0.5) / (df + 0.5))
                    self._idf_diag = sp.spdiags(bm25idf,
                                                diags=0, m=n_features, n=n_features)

            # Vanilla idf
            elif self.delta_idf:
                if y is None:
                    raise ValueError("Labels are needed to determine Delta idf")

                N1, df1, N2, df2 = _class_frequencies(X, y)
                delta_idf = np.log((df1 * float(N2) + int(self.smooth_idf)) /
                                   (df2 * N1 + int(self.smooth_idf)))

                # Maybe scale delta_idf to only positive values (for Naive Bayes, etc) ?
                self._idf_diag = sp.spdiags(delta_idf,
                                            diags=0, m=n_features, n=n_features)

            else:
                df = _document_frequency(X)

                df += int(self.smooth_idf)
                n_samples += int(self.smooth_idf)

                idf = np.log(float(n_samples) / df) + 1.0
                self._idf_diag = sp.spdiags(idf,
                                            diags=0, m=n_features, n=n_features)

        return self

    def transform(self, X, copy=True):

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csr_matrix(X, copy=copy)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.bm25_tf:

            D = (X.sum(1) / np.average(X.sum(1))).reshape((n_samples, 1))
            D = ((1 - self.b) + self.b * D) * self.k

            D_X = _add_sparse_column(X, D)

            X.data = np.divide(X.data * (self.k + 1), D_X.data, X.data)

        elif self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            if not hasattr(self, "_idf_diag"):
                raise ValueError("idf vector not fitted")
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
                                     
            X = np.dot(X, self._idf_diag)

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None


def _add_sparse_column(sparse, column):
    addition = sp.lil_matrix(sparse.shape)
    sparse_coo = sparse.tocoo()
    for i, j, v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
        addition[i, j] = v + column[i, 0]
    return addition.tocsr()


def _class_frequencies(X, y):

    labels = np.unique(y)
    if len(labels) > 2:
        raise ValueError("Delta works only with binary classification problems")

    N1 = np.where(y == labels[0])[0]
    N2 = np.where(y == labels[1])[0]

    df1 = np.bincount(X[N1].nonzero()[1], minlength=X.shape[1])
    df2 = np.bincount(X[N2].nonzero()[1], minlength=X.shape[1])

    return N1.shape[0], df1, N2.shape[0], df2


def _document_frequency(X):
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


# nltk.download('wordnet')
# nltk.download('stopwords')
# demoji.download_codes()
warnings.filterwarnings("ignore")

data = pd.read_csv('./Final_represent/tweet_data_final.txt')

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
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

def emojitotext(text):
    # return text
    emoji = demoji.findall(text)
    for emojicode, word in emoji.items():
        text = text.replace(emojicode, " " + word)
    return text

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

tfidf_vectorizer = CountVectorizer(
    min_df=1, max_features=720,
    strip_accents='unicode', analyzer='word',
    ngram_range=(1, 1), lowercase=True,
    input='content', encoding='utf-8',
    decode_error='strict', preprocessor=None,
    tokenizer=None, stop_words=None,
    token_pattern=r"\w+", max_df=1.0,
    vocabulary=None, binary=False, dtype=np.float64
)

bm25 = BM25Transformer(use_idf=True, use_bm25idf=True)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text_parse'])

bm25.fit(X_train_tfidf, y_train)
X_train_tfidf = bm25.transform(X_train_tfidf)

X_test_tfidf = tfidf_vectorizer.transform(X_test['text_parse'])
X_test_tfidf = bm25.transform(X_test_tfidf)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)
predictions = naive_bayes.predict(X_test_tfidf)

# print(classification_report(y_test, predictions,
#                             target_names=category_codes.keys()))

cm = confusion_matrix(y_test, predictions)
row_sums = cm.sum(axis=1)
cm = cm / row_sums[:, np.newaxis]

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

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
peraccuracy = (TP + TN) / (TP + FP + FN + TN)

perprecision = (TP) / (TP + FP)
perrecall = TP / (TP + FN)
perf1 = 2 * perprecision * perrecall / (perprecision + perrecall)

columns = ['accuracy', 'precision', 'recall', 'F1']
finalmetrics = pd.DataFrame(index=category_codes.keys(), columns=columns)
finalmetrics = finalmetrics.fillna(0)

finalmetrics['accuracy'] = peraccuracy
finalmetrics['precision'] = perprecision
finalmetrics['recall'] = perrecall
finalmetrics['F1'] = perf1
finalmetrics = finalmetrics.round(2)
print(finalmetrics)

# plt.figure(figsize=(6, 4))
# plt.plot(perf1, 'o-', color='b', linewidth=4)
# plt.title("BM25 F1")
# tick_marks = np.arange(len(category_codes.keys()))
# plt.xticks(tick_marks, category_codes.keys(), rotation=60, fontsize=8)
# plt.yticks(fontsize=8)
# plt.ylabel("F1", fontsize=12)
# plt.show()

plot_confusion_matrix(cm, classes=category_codes.keys(), normalize=True,
                      title='confusion matrix')
