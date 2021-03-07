import numpy as np
import math
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity

snowball_stemmer = SnowballStemmer('english')

class TfIdf:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.documents = {}
        self.docidf = {} # corpus dict
        self.sims2 = {} # the ans of similirity

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            w = snowball_stemmer.stem(w) # 詞性還原
            if w not in self.stop_words:
                #TF
                doc_dict[w] = doc_dict.get(w, 0.) + 1.0

        for wc in doc_dict:
            # IDF preprocessing
            self.docidf[wc] = self.docidf.get(wc, 0.0)+1.0
            # TF ---- Log Normalization
            doc_dict[wc] = 1+math.log(doc_dict.get(wc, 0.0),2)

        # add the document to the corpus  (TF)
        self.documents[doc_name] = doc_dict

    def similarities(self, queryName, list_of_words):

        query_dict = {}

        for w in list_of_words:
            w = snowball_stemmer.stem(w)  
            query_dict[w] = query_dict.get(w, 0.0) + 1.0
        
        #Double normalization σ
        maxquery = max(query_dict.items(), key=operator.itemgetter(1))[1]

        for wc in query_dict:
            query_dict[wc] = 0.8 + query_dict.get(wc, 0.0)*0.2/maxquery

        # computing the list of similarities
        scoreDic = {}
        for doc in self.documents:
            docTFIDF = []
            qTFIDF = []
            # lengthDoc = 0
            # lengthQuery = 0
            # score = 0
            dicTemp = self.documents[doc]

            # TFIDF
            for w in self.docidf:
                if w in dicTemp:
                    docTFIDF.append(
                        dicTemp[w]*math.log10((1+4191)/(self.docidf[w]+1)))
                else:
                    docTFIDF.append(0)

                if w in query_dict:
                    qTFIDF.append(
                        (1+query_dict[w])*math.log10((1+4191)/(self.docidf[w]+1)))
                else:
                    qTFIDF.append(0)

            # cosine_similarity
            arrayQuery = np.array(qTFIDF)
            arrayDoc = np.array(docTFIDF)

            scoreDic[doc] = cosine_similarity([arrayQuery], [arrayDoc])
            # lengthDoc = np.sqrt(arrayDoc.dot(arrayDoc))
            # lengthQuery = np.sqrt(arrayQuery.dot(arrayQuery))

            # score = arrayQuery.dot(arrayDoc)/(lengthDoc*lengthQuery)

            # scoreDic[doc] = score

        # 排序
        self.sims2[queryName] = sorted(
            scoreDic.items(), key=lambda d: d[1], reverse=True)
