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

class RocchioAlgo:
    def __init__(self,rank_amount):
        self.stop_words = set(stopwords.words('english'))
        self.doc_name = []
        self.query_name = []
        # self.document = {}
        self.document = []
        # self.query = {}
        self.query = []
        self.rank_amount = rank_amount
        self.docidf = {}
        self.querryidf = {}
        self.ans = []

    def readfile(self):
        global root_path
        #read doc
        with open(root_path + 'doc_list.txt', 'r') as L:
            for filename in tqdm(L):
                filename = filename.strip('\n')
                path = root_path + 'docs/' 
                file = path+filename+".txt"
                self.doc_name.append(filename)
                with open(file, 'r') as f:
                    listOfWords = []
                    for lines in f.readlines():
                        listOfWords = nltk.word_tokenize(lines)
                        f.close()
                    
                    doc_dict = {}
                    for w in listOfWords:
                        w = snowball_stemmer.stem(w)  # part of speech stem
                        if w not in self.stop_words:
                            doc_dict[w] = doc_dict.get(w, 0.) + 1.0 # TF

                    for wc in doc_dict:
                        # IDF preprocessing
                        self.docidf[wc] = self.docidf.get(wc, 0.0)+1.0
                        # TF ---- Log Normalization
                        doc_dict[wc] = 1 + math.log(doc_dict.get(wc, 0.0), 2)
                    
                    self.document.append(doc_dict)

        #read query
        with open(root_path + 'query_list.txt', 'r') as Q:
            for queryfilename in tqdm(Q):
                queryfilename = queryfilename.strip('\n')
                path = root_path + 'queries/'
                file = path+queryfilename+".txt"
                self.query_name.append(queryfilename)
                with open(file, 'r') as f:
                    listOfWords = []
                    for lines in f.readlines():
                        listOfWords = nltk.word_tokenize(lines)
                        f.close()
                    
                    query_dict = {}
                    for w in listOfWords:
                        w = snowball_stemmer.stem(w) 
                        if w not in self.stop_words:
                            query_dict[w] = (query_dict.get(w, 0.) + 1.0) # TF
                    
                    for wc in query_dict:
                        # IDF preprocessing
                        self.querryidf[wc] = self.querryidf.get(wc, 0.0)+1.0

                    self.query.append(query_dict)

        print('read file down')

    def pseudo_read(self,pseudo_docname):
        pseudo_list = []
        with open(pseudo_docname, 'r') as pseudo_file:
            for line in pseudo_file:
                if line == 'Query,RetrievedDocuments\n':
                    continue
                pseudo_name = re.split(',| |\n', line)
                pseudo_name.pop()
                pseudo_name.pop()
                pseudo_name.pop(0)
                pseudo_list.append(pseudo_name)
        return pseudo_list

    def pseudo_relevant_doc(self,first):
        relevant_doc_word = []

        for d_list in first:
            total_voc = {}
            for d_name in d_list:
                temp = copy.deepcopy(self.document[self.doc_name.index(d_name)])
                for temp_word in temp:
                    if temp_word not in total_voc:
                        total_voc[temp_word] = temp[temp_word]
                    else:
                        total_voc[temp_word] += temp[temp_word]

            relevant_doc_word.append(total_voc)     # query k relevant doc all word

        return relevant_doc_word

    def pseudo_non_relevant_doc(self,first):
        nonrelevant_doc_word = []

        for d_list in first:
            total_voc = {}
            for d_name in d_list:
                temp = copy.deepcopy(self.document[self.doc_name.index(d_name)])
                for temp_word in temp:
                    if temp_word not in total_voc:
                        total_voc[temp_word] = temp[temp_word]
                    else:
                        total_voc[temp_word] += temp[temp_word]

            nonrelevant_doc_word.append(total_voc) 

        return nonrelevant_doc_word

    def rocchioalgor(self, alpha, beta, gamma, relevant_doc, file_num, non_rel_doc, non_file_num):
        Doc_tfidf, Q_tfidf, rel_tfidf, non_rel_tfidf = self.tf_idf_with_rel(relevant_doc, non_rel_doc)
        print('tfidf down')

        start = time.time()
        self.ans = self.ROCCHIO(Doc_tfidf, Q_tfidf, rel_tfidf, non_rel_tfidf, alpha, beta, gamma, file_num, non_file_num)
        print('rocchio down')
        print(time.time() - start)

    def tf_idf_with_rel(self, relevant_doc, non_rel_doc):
        N = len(self.doc_name)+1
        Doc_idf = self.docidf.copy()
        x = list(Doc_idf.keys())
        for i in range(0,len(Doc_idf)-1):
            Doc_idf[x[i]] = math.log10(N/(Doc_idf[x[i]]+1)) #idf 

        Doc_tfidf = copy.deepcopy(self.document)
        for j in range(0,len(Doc_tfidf)-1):
            x = list(self.document[j].keys())
            for i in range(0,len(x)-1):
                Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

        Q_tfidf = copy.deepcopy(self.query)

        for j in range(0, len(Q_tfidf)-1):

            x = list(self.query[j].keys())

            for i in range(0, len(x)-1):
                if(x[i] in Doc_idf):
                    Q_tfidf[j][x[i]] = self.query[j][x[i]]*Doc_idf[x[i]] #tf*idf
                else:
                    Q_tfidf[j][x[i]] = 0

        rel_tfidf = copy.deepcopy(relevant_doc)
        for j in range(0,len(rel_tfidf)-1):

            x = list(relevant_doc[j].keys())

            for i in range(0,len(x)-1):
                if(x[i] in Doc_idf):
                    rel_tfidf[j][x[i]] = relevant_doc[j][x[i]]*Doc_idf[x[i]] #tf*idf
                else:
                    rel_tfidf[j][x[i]] = 0
        
        non_rel_tfidf = copy.deepcopy(non_rel_doc)
        for j in range(0,len(rel_tfidf)-1):

            x = list(non_rel_doc[j].keys())

            for i in range(0,len(x)-1):
                if(x[i] in Doc_idf):
                    non_rel_tfidf[j][x[i]] = non_rel_doc[j][x[i]]*Doc_idf[x[i]] #tf*idf
                else:
                    non_rel_tfidf[j][x[i]] = 0

        return Doc_tfidf, Q_tfidf, rel_tfidf, non_rel_tfidf

    def ROCCHIO(self, Doc_tfidf, Q_tfidf, rel_tfidf, non_rel_tfidf, alpha, beta, gamma, file_num, non_file_num):

        Ans_T = []
        for q, old_que_dic in enumerate(Q_tfidf):
            if q % 50 == 0:
                print(time.strftime("%D,%H:%M:%S"))
                print(q)
            Sim = []

            que_dic = copy.deepcopy(old_que_dic)  # new_q
            for w in que_dic:
                que_dic[w] *= alpha
            for rel in rel_tfidf[q]:
                if rel in que_dic:
                    que_dic[rel] += beta * rel_tfidf[q][rel] / file_num
                else:
                    que_dic[rel] = beta * rel_tfidf[q][rel] / file_num
            
            # for nonrel in non_rel_tfidf[q]:
            #     if nonrel in que_dic:
            #         que_dic[nonrel] += gamma * non_rel_tfidf[q][nonrel] / non_file_num
            #     else:
            #         que_dic[nonrel] = gamma * non_rel_tfidf[q][nonrel] / non_file_num
            
            #cosine_similarity
            for doc_dic in Doc_tfidf:
                a, b = 0, 0
                for que_voc in que_dic:
                    if que_dic[que_voc] == 0:
                        continue
                    if que_voc in doc_dic:
                        a += que_dic[que_voc] * doc_dic[que_voc]    # 被除數
                    b += pow(que_dic[que_voc], 2)    # 除數1

                c = sum(pow(doc_dic[doc_voc], 2) for doc_voc in doc_dic)  # 除數2

                Sim.append(a / (math.sqrt(b)*math.sqrt(c)))

            # arrayQuery = np.array(que_dic)
            # arrayDoc = np.array(Doc_tfidf)

            # Sim.append( cosine_similarity([arrayQuery], [arrayDoc]) )

            Sim_sort = sorted(Sim, reverse=True)

            Ans = []

            for i in range(0, self.rank_amount):
                Ans.append(self.doc_name[Sim.index(Sim_sort[i])] )
            Ans_T.append(Ans)

        return Ans_T

    def writeAns(self, file_name):
        with open(str(file_name) + '.txt', 'w') as file:
            file.write("Query,RetrievedDocuments\n")
            for i in range(0, len(self.query_name)):
                file.write(str(self.query_name[i]) + ',')
                for num, j in enumerate(self.ans[i]):
                    if num < self.rank_amount:
                        file.write(str(j) + ' ')
                    else:
                        break
                file.write('\n')


rocchio_cal = RocchioAlgo(5001) # rank_amount
rocchio_cal.readfile()
pseudodoc = rocchio_cal.pseudo_read('D:/Python_test/IR/bm25_relevant_10_0.07.txt')
pseudonondoc = rocchio_cal.pseudo_read('D:/Python_test/IR/bm25_non_relevant_2.txt')
new_que = rocchio_cal.pseudo_relevant_doc(pseudodoc)
non_rel = rocchio_cal.pseudo_non_relevant_doc(pseudonondoc)
rocchio_cal.rocchioalgor(1, 0.2, 0, new_que, 10, non_rel, 0) #relevant doc count is 10
rocchio_cal.writeAns('rocchio_final')