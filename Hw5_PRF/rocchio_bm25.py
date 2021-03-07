import os
import math
import copy
import re
import time

import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

snowball_stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

root_path = 'D:/Python_test/IR/Hw5_PRF/ir_hw5_data/'

# bm25 parameters
K3 = 0.2  # query k
K1 = 0.8  # doc k
# K1 = 0.0024
b = 0.7
# b = 0.0015
# K2 = 0.02
K2 = 0.03
# delta = 0.4
delta = 0.5

alldoclen = []
qalldoclen = []

i = 0


class Rocchio_bm25:
    def __init__(self, rank_amount):
        self.stop_words = set(stopwords.words('english'))+['\n']
        self.doc_name = []
        self.query_name = []
        self.documents = []
        self.queries = []
        self.alldoclen = []
        self.avgdoclen = 0.0
        self.allquerlen = []
        self.rank_amount = rank_amount
        self.docidf = {}
        self.querryidf = {}
        self.ans = []
        self.sim_ans = {}

    def doctf_readfile(self):
        # read doc
        global root_path
        with open(root_path + 'doc_list.txt', 'r') as L:
            for filename in tqdm(L):
                tempdoclen = 0
                filename = filename.strip('\n')
                path = root_path + 'docs/'
                file = path+filename+".txt"
                self.doc_name.append(filename)
                with open(file, 'r') as f:
                    listOfWords = []
                    for lines in f.readlines():
                        listOfWords = nltk.word_tokenize(lines)
                        # self.alldoclen.append(len(listOfWords))
                        f.close()

                    doc_dict = {}
                    for w in listOfWords:
                        w = snowball_stemmer.stem(w)  # part of speech stem
                        if w not in self.stop_words:
                            tempdoclen += 1
                            doc_dict[w] = doc_dict.get(w, 0.) + 1.0  # TF

                    self.alldoclen.append(tempdoclen)

                    for wc in doc_dict:
                        # IDF preprocessing
                        self.docidf[wc] = self.docidf.get(wc, 0.0)+1.0
                        # TF ---- Log Normalization
                        doc_dict[wc] = 1 + math.log(doc_dict.get(wc, 0.0), 2)

                    self.documents.append(doc_dict)
                    # self.documents[filename] = doc_dict

        tempalldoclen = np.array(self.alldoclen)
        self.avgdoclen = np.mean(tempalldoclen[np.nonzero(tempalldoclen)])

        print('read doc file down')

    def querytf_readfile(self):
        # read query
        global root_path
        with open(root_path + 'query_list.txt', 'r') as Q:
            for queryfilename in tqdm(Q):
                tempquerlen = 0
                queryfilename = queryfilename.strip('\n')
                path = root_path + 'queries/'
                file = path+queryfilename+".txt"
                self.query_name.append(queryfilename)
                with open(file, 'r') as f:
                    listOfWords = []
                    for lines in f.readlines():
                        listOfWords = nltk.word_tokenize(lines)
                        # self.allquerlen.append(len(listOfWords))
                        f.close()

                    query_dict = {}
                    for w in listOfWords:
                        w = snowball_stemmer.stem(w)
                        if w not in self.stop_words:
                            tempquerlen += 1
                            query_dict[w] = (query_dict.get(w, 0.) + 1.0)  # TF

                    self.allquerlen.append(tempquerlen)

                    for wc in query_dict:
                        # IDF preprocessing
                        self.querryidf[wc] = self.querryidf.get(wc, 0.0)+1.0

                    self.queries.append(query_dict)
                    # self.queries[queryfilename] = query_dict

        print('read query file down')

    def pseudo_read(self, pseudo_docname):
        pseudo_list = []
        with open(pseudo_docname, 'r') as pseudo_file:
            for line in pseudo_file:
                if line == 'Query,RetrievedDocuments\n':
                    continue
                pseudo_name = re.split(',| |\n', line)
                pseudo_name.pop()
                pseudo_name.pop()
                # pseudo_name.remove('\n')
                pseudo_name.pop(0)
                pseudo_list.append(pseudo_name)
        return pseudo_list

    def pseudo_relevant_doc(self, first):
        # relevant_doc = []
        relevant_doc_word = []

        for d_list in first:
            # doc_dict = []
            total_voc = {}
            for d_name in d_list:
                temp = copy.deepcopy(
                    self.documents[self.doc_name.index(d_name)])
                # print(temp)
                # doc_dict.append(temp)
                for temp_word in temp:
                    if temp_word not in total_voc:
                        # total_voc[temp_word] = temp[temp_word]
                        total_voc[temp_word] = 1
                    else:
                        # total_voc[temp_word] += temp[temp_word]
                        total_voc[temp_word] += 1

            for temp_word in total_voc:
                total_voc[temp_word] = 1 + math.log(total_voc.get(temp_word, 0.0), 2)
                    # if temp_word not in self.querryidf:
                    #     self.querryidf[temp_word] = self.querryidf.get(
                    #         temp_word, 0.0)+1.0

            # query k relevant doc all word
            relevant_doc_word.append(total_voc)
            # print(relevant_doc_word)
            # break

        return relevant_doc_word

    def Rocchio_BM25(self, K3, b, K1, K2, delta, alpha, beta, gamma, rel_doc, non_rel_doc, reldocnum, nonreldocnum):
        # j = 0
        for j, query in tqdm(enumerate(self.queries)):
            simscore_dic = {}
            queryName = self.query_name[j]
            # qlen = self.allquerlen[j]
            relevantdoc = rel_doc[j]
            # j += 1
            # i = -1
            tempqueryidf = dict(query)
            tempqueryidf.update(relevantdoc)
            qlen = sum(tempqueryidf.values())
            for i, doc in enumerate(self.documents):
                # i = i+1
                dlen = self.alldoclen[i]
                docname = self.doc_name[i]
                score = 0.0
                for w in tempqueryidf.keys():
                    if w in doc:
                        ctd = 0.8*doc[w]/((1-b)+b*dlen/self.avgdoclen)
                        first = ((K1 + 1) * (ctd+delta)) / (K1+ctd*0.7)
                    else:
                        first = 0.0

                    if w in query:
                        second = alpha*((K3+1) * query[w]) / (K3 + query[w])
                    else:
                        second = 0.0

                    if w in relevantdoc:
                        second2 = beta *((K3+1) * relevantdoc[w]) / (K3 + relevantdoc[w])
                    else:
                        second2 = 0.0

                    score += first *(second+second2)*math.log10((30000 - self.docidf[w]+0.5)/(self.docidf[w]+0.5))

                bm11 = K2*qlen*abs(self.avgdoclen-dlen)/(self.avgdoclen+dlen)
                score += bm11

                simscore_dic[docname] = score

            # 排序
            self.sim_ans[queryName] = sorted(
                simscore_dic.items(), key=lambda d: d[1], reverse=True)
            # self.sim_ans[queryName] = sorted(simscore_dic.items(), key=lambda d: d[1])

    def writeAns(self, file_name):
        ans = "Query,RetrievedDocuments"
        for Queryname in self.sim_ans:
            i = 1
            ans += "\n"+Queryname+","
            for key, value in self.sim_ans[Queryname]:
                if i == self.rank_amount:
                    break
                else:
                    ans += key+" "
                    i += 1
        ans += "\n"
        bm25_file = open(file_name, "w")
        bm25_file.write(ans)
        bm25_file.close()


rocchio_cal = Rocchio_bm25(5000)  # rank_amount
rocchio_cal.doctf_readfile()
rocchio_cal.querytf_readfile()

pseudodoc = rocchio_cal.pseudo_read('D:/Python_test/IR/bm25_relevant_5.txt')
pseudonondoc = rocchio_cal.pseudo_read(
    'D:/Python_test/IR/bm25_non_relevant_2.txt')
new_que = rocchio_cal.pseudo_relevant_doc(pseudodoc)
non_rel = rocchio_cal.pseudo_relevant_doc(pseudonondoc)
rocchio_cal.Rocchio_BM25(K3, b, K1, K2, delta, 1, 1, 0,
                         new_que, non_rel, 5, 0)  # relevant doc count is 5
rocchio_cal.writeAns('test_rocchio_bm25.txt')
