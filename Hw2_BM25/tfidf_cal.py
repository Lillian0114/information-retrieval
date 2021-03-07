import numpy as np
import math
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
# import statistics

snowball_stemmer = SnowballStemmer('english')

class TfIdf:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.documents = {}  # 每篇doc的tf
        self.queries = {}  # 每個que的tf
        self.docidf = {}  # doc idf
        self.queidf = {} #corpus_dict
        self.sim_ans = {}
        self.othercorpus = []

    def doc_tf(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            w = snowball_stemmer.stem(w)  # 詞性還原
            if w not in self.stop_words:
                # TF
                doc_dict[w] = (doc_dict.get(w, 0.) + 1.0)
        
        # try:
            # avgtf = statistics.mean(doc_dict.values()) 
            # maxquery = max(doc_dict.items(), key=operator.itemgetter(1))[1]
        # except:
            # print(doc_name)

        for wc in doc_dict:
            # IDF preprocessing
            self.docidf[wc] = self.docidf.get(wc, 0.0)+1.0
            # TF ---- Log Normalization
            doc_dict[wc] = (1 + math.log(doc_dict.get(wc, 0.0), 2))
            # doc_dict[wc] = (1 + math.log(doc_dict.get(wc, 0.0), 2))/(1+math.log(avgtf,2))
            # doc_dict[wc] = 0.4 + doc_dict.get(wc, 0.0)*0.6/maxquery

        # add the document to the corpus  (TF)
        self.documents[doc_name] = doc_dict
    
    def query_tf(self, queryName, list_of_words):
        query_dict = {}
        for w in list_of_words:
            w = snowball_stemmer.stem(w)
            if w not in self.stop_words:
                query_dict[w] = (query_dict.get(w, 0.0) + 1 )

        for wc in query_dict:
            self.queidf[wc] = self.queidf.get(wc, 0.0)+1.0
            if wc not in self.docidf:
                self.docidf[wc] = self.docidf.get(wc, 0.0)

        self.queries[queryName] = query_dict

    def BM25(self, K1, b, avglen, alllen, K3, queryName,K2,qalllen):
        simscore_dic = {}
        i=-1
        first=0.0
        second=0.0
        # delta = 0.09
        delta = 0.5
        for doc in self.documents:
            i=i+1
            len=alllen[i]
            score = 0.0
            docTFtemp = self.documents[doc]
            for w in self.queidf:
                if w in docTFtemp:
                    ctd = 0.765*docTFtemp[w]/((1-b)+b*len/avglen)
                    first = ((K1 + 1) * (ctd+delta)) / (K1+ctd*0.7)
                    # first = ((K1 + 1) * 0.765*docTFtemp[w]) / ((K1*((1-b)+b*len/avglen))+docTFtemp[w])
                else:
                    first = 0.0
                if w in self.queries[queryName]:
                    second = ((K3+1) *self.queries[queryName][w]) / (K3 + self.queries[queryName][w])
                else:
                    second = 0.0

                score += first*second*math.log10((4191-self.docidf[w]+0.5)/(self.docidf[w]+0.5))

            bm11 = K2*qalllen*abs(avglen-len)/(avglen+len)
            score += bm11

            simscore_dic[doc] = score
            
        # 排序
        self.sim_ans[queryName] = sorted(simscore_dic.items(), key=lambda d: d[1], reverse=True)

