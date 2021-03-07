import os
import re
import time
import math
import copy
# from rocchio_test import RocchioAlgo
import numpy as np

import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

root_path = 'D:/Python_test/IR/Hw5_PRF/ir_hw5_data/'

QUERY = []
DOCUMENT = []
# BG = []
BG = {}
DOC_NAME = []
QUERY_NAME = []

# ROCCHIO_ALPHA = 0.7
# ROOCHIO_BETA = 0.3
TMM_ALPHA = 0.3
TMM_BETA = 0.2
KL_A = 0.4
KL_B = 0.3
KL_D = 0.7

EMtimes = 6

RANKING = []

def readfile():
    global root_path
    global QUERY, DOCUMENT, BG
    global QUERY_NAME, DOC_NAME

    #read doc
    with open(root_path + 'doc_list.txt', 'r') as L:
        for filename in tqdm(L):
            filename = filename.strip('\n')
            path = root_path + 'docs/' 
            file = path+filename+".txt"
            DOC_NAME.append(filename)
            with open(file, 'r') as f:
                listOfWords = []
                for lines in f.readlines():
                    listOfWords = nltk.word_tokenize(lines)
                    f.close()
                
                doc_dict = {}
                for w in listOfWords:
                    w = snowball_stemmer.stem(w)  # part of speech stem
                    if w not in stop_words:
                        doc_dict[w] = doc_dict.get(w, 0.) + 1.0 # TF
                
                DOCUMENT.append(doc_dict)
                # DOCUMENT[filename] = doc_dict

    #read query
    with open(root_path + 'query_list.txt', 'r') as Q:
        for queryfilename in tqdm(Q):
            queryfilename = queryfilename.strip('\n')
            path = root_path + 'queries/'
            file = path+queryfilename+".txt"
            QUERY_NAME.append(queryfilename)
            with open(file, 'r') as f:
                listOfWords = []
                for lines in f.readlines():
                    listOfWords = nltk.word_tokenize(lines)
                    f.close()
                
                query_dict = {}
                for w in listOfWords:
                    w = snowball_stemmer.stem(w) 
                    if w not in stop_words:
                        query_dict[w] = (query_dict.get(w, 0.) + 1.0) # TF

                QUERY.append(query_dict)
                # QUERY[queryfilename] = query_dict
    
    # Load BG
    # with open('BGLM_hw5_unique.txt') as BG_file:
    #     for len_voc_f, val_voc in enumerate(BG_file):
    #         bg_split = re.split(' |\n', val_voc)
    #         BG.append(float(bg_split[1]))
    with open("D:/Python_test/IR/HW5_PRF/BGLM_hw5_unique.txt", 'r') as BG_file:
        for val_voc in BG_file:
            bg_split = re.split(' ', val_voc)
            BG[bg_split[0]] = float(bg_split[1])
            # break

    print('read file down')

def ans_read(ans):
    ans_list = []
    with open(ans) as ans_file:
        for line in ans_file:
            if line == 'Query,RetrievedDocuments\n':
                continue
            ans_name = re.split(',| ', line)
            ans_name.remove('\n')
            ans_name.pop(0)
            ans_list.append(ans_name)
            
    return ans_list

def TMM(relevantdoc): 
    global TMM_ALPHA, TMM_BETA
    global DOCUMENT, DOC_NAME, BG
    relevant_doc = []
    relevant_doc_word = []

    for d_list in relevantdoc:
        doc_dict = []
        total_voc = {}
        for d_name in d_list:
            temp = copy.deepcopy(DOCUMENT[DOC_NAME.index(d_name)])
            doc_dict.append(temp)
            for temp_word in temp:
                if temp_word not in total_voc:
                    total_voc[temp_word] = temp[temp_word]
                else:
                    total_voc[temp_word] += temp[temp_word]

        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    tmm_list = copy.deepcopy(relevant_doc_word)

    for iteration in range(1, EMtimes):
        print(iteration)
        for q in range(len(relevant_doc_word)):  # query 150
            tmm = copy.deepcopy(tmm_list[q])
            if iteration == 1:
                # for tmm_voc in tmm:
                #     tmm[tmm_voc] = random.random()
                tmm_total = sum(tmm.values())
                for tmm_voc in tmm:
                    tmm[tmm_voc] /= tmm_total
            tmmwd = copy.deepcopy(relevant_doc[q])  # 只是剛好relevant_doc[q] = q all file dictionary
            tdwd = copy.deepcopy(relevant_doc[q])   # 同上

            # E_Step
            for doc_id, doc in enumerate(relevant_doc[q]):
                doc_total = sum(doc.values())
                for word_id, word in enumerate(doc):
                    pbg = BG.get(word,0.)
                    pwd = doc[word] / doc_total
                    # if int(word) < len(BG):
                    #     pbg = math.exp(BG[int(word)])
                    twd_deno = ((tmm[word] * (1 - TMM_ALPHA - TMM_BETA)) + (pwd * TMM_ALPHA) + (pbg * TMM_BETA))
                    tmmwd[doc_id][word] = (tmm[word] * (1 - TMM_ALPHA - TMM_BETA)) / twd_deno
                    tdwd[doc_id][word] = (pwd * TMM_ALPHA) / twd_deno
                    # twd_deno = np.logaddexp( np.logaddexp(np.log(1 - TMM_ALPHA - TMM_BETA)+np.log(tmm[word]),np.log(TMM_ALPHA)+np.log(pwd) ),np.log(TMM_BETA)+np.loh(pbg) )
                    # tmmwd[doc_id][word] = np.log(1-TMM_ALPHA-TMM_BETA)+np.log(tmm[word])-twd_deno
                    # tdwd[doc_id][word] = np.log(pwd) +np.log(TMM_ALPHA) - np.log(twd_deno)

            # M_Step
            tmm_molecular_list = []
            for word_id, word in enumerate(tmm):
                molecular = sum(tmmwd[doc_id][word] * doc[word] for doc_id, doc in enumerate(relevant_doc[q]) if word in doc)
                tmm_molecular_list.append(molecular)
            denominator = sum(tmm_molecular_list)

            for word_id, word in enumerate(tmm):
                tmm[word] = tmm_molecular_list[word_id] / denominator
            tmm_list[q] = tmm

            for doc_id, doc in enumerate(relevant_doc[q]):
                td_molecular_list = []
                for word in doc:
                    molecular = tdwd[doc_id][word] * doc[word]
                    td_molecular_list.append(molecular)
                denominator = sum(td_molecular_list)
                for word_id, word in enumerate(doc):
                    relevant_doc[q][doc_id][word] = td_molecular_list[word_id] / denominator

            # print(sum(tmm.values()))
            # print(sum(relevant_doc[q][0].values()))

        #loglikelihood
        l = 1
        for doc_id, doc in enumerate(relevant_doc[0]):
            word_total = sum(doc.values())
            for word_id, word in enumerate(doc):
                # np.sum(np.sum(np.logaddexp(np.log(1 - self._lambda)+self.log_Psmm_matix, np.log(self._lambda)+self.log_BG_matix).T*np.exp(self.Rel_count_query_matrix), axis = 1),axis = 0)
                l += math.log10(pow(((1 - TMM_ALPHA - TMM_BETA) * tmm_list[0][word]) + (TMM_ALPHA * doc[word]) + (TMM_BETA * BG.get(word,0.) ), doc[word]))
                # l += math.log10(pow(((1 - TMM_ALPHA - TMM_BETA) * tmm_list[0][word]) + (TMM_ALPHA * doc[word]) + (TMM_BETA * BG.get(word,0.) ), doc[word]))
                # l += math.log10(pow(((1 - TMM_ALPHA - TMM_BETA) * tmm_list[0][word]) + (TMM_ALPHA * doc[word]) + (TMM_BETA * math.exp(BG[int(word)]) ), doc[word]))
        print(math.exp(l))

    return tmm_list
    # return 0

def KL(tmm_query):
    global QUERY, DOCUMENT, RANKING, DOC_NAME,BG
    global KL_A, KL_B, KL_D
    for q_id, q in enumerate(QUERY):
        if q_id % 50 == 0:
            print(q_id)
            print(time.strftime("%D,%H:%M:%S"))
        kl_list = []
        tmq = copy.deepcopy(tmm_query[q_id])

        q_total = sum(q.values())
        new_q = copy.deepcopy(q)
        new_q.update(tmq)
        
        #KL
        for doc_id, doc in enumerate(DOCUMENT):
            doc_total = sum(doc.values())
            kl_score = 0
            for q_word in new_q:
                if doc_id == 0:
                    base_q, tmm, pbg = 0, 0, 0
                    new_q[q_word] = 0

                    if q_word in tmq:
                        tmm = tmq[q_word]
                    if q_word in q:
                        base_q = q[q_word]

                    pbg = BG.get(q_word,0.)
                    # if int(q_word) < len(BG):
                    #     pbg = math.exp(BG[int(q_word)])

                    new_q[q_word] = (KL_A * base_q / q_total) + (KL_B * tmm) + ((1 - KL_A - KL_B) * pbg)

                if q_word in doc:
                    new_doc = (KL_D * doc[q_word] / doc_total) + ((1 - KL_D) * BG.get(q_word,0.) )
                    # new_doc = (KL_D * doc[q_word] / doc_total) + ((1 - KL_D) * math.exp(BG[int(q_word)]))
                    kl_score -= (new_q[q_word]) * math.log10(new_doc)
                else:
                    new_doc = ((1 - KL_D) * BG.get(q_word,0.) )
                    kl_score -= (new_q[q_word]) * math.log10(new_doc)
            
            kl_list.append(kl_score)

        #sorting
        sort = sorted(kl_list)
        q_rank = []
        for sort_num in sort:
            q_rank.append(DOC_NAME[kl_list.index(sort_num)])

        RANKING.append(q_rank)

def writefile(name):
    global QUERY_NAME, DOC_NAME, RANKING
    with open(name + '.txt', 'w') as retrieval_file:
        retrieval_file.write("Query,RetrievedDocuments\n")
        for retrieval_id, retrieval_list in enumerate(RANKING):
            retrieval_file.write(QUERY_NAME[retrieval_id] + ',')
            for retrieval_name in retrieval_list[0:5001]:
                retrieval_file.write(retrieval_name + ' ')
            if retrieval_id != len(QUERY_NAME) - 1:
                retrieval_file.write('\n')


readfile()
# VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=1)
# VSM_re.calculate()
# VSM_re.writeAns('VSM5')
# print('VSM down')
answer = ans_read('bm25_relevant_1.txt')
tmm_query = TMM(answer)
print('TMM down')
KL(tmm_query)
writefile('tmm_ggggg')
print('KL down')

# QL(tmm_query)
# writefile('QL')
# print('QL down')

# print('Process down')



'''
def QL(tmm_query):
    global QUERY, DOCUMENT, RANKING, DOC_NAME, BG
    global KL_A, KL_B, KL_D

    for q_id, q in enumerate(QUERY):
        if q_id % 100 == 0:
            print(q_id)

        tmq = copy.deepcopy(tmm_query[q_id])
        q_total = sum(q.values())
        new_q = copy.deepcopy(q)
        new_q.update(tmq)

        for q_word in new_q:
            base_q, tmm, pbg = 0, 0, 0
            new_q[q_word] = 0

            if q_word in tmq:
                tmm = tmq[q_word]
            if q_word in q:
                base_q = q[q_word]
            if int(q_word) < len(BG):
                pbg = math.exp(BG[int(q_word)])
            new_q[q_word] = (KL_A * base_q / q_total) + (KL_B * tmm) + ((1 - KL_A - KL_B) * pbg)

        score_list = []
        for doc in DOCUMENT:
            doc_total = sum(doc.values())
            score = 1
            for q_word in new_q:
                if q_word in doc:
                    score *= ((0.7 * doc[q_word] / doc_total) + (0.3 * math.exp(BG[int(q_word)])))
            score_list.append(score)

        sort = sorted(score_list, reverse=True)
        q_rank = []
        for sort_num in sort:
            q_rank.append(DOC_NAME[score_list.index(sort_num)])

        RANKING.append(q_rank)

def relevant_doc(first):
    global QUERY
    relevant_doc = []
    relevant_doc_word = []
    new_q = []

    for d_list in first:
        doc_dict = []
        total_voc = {}
        for d_name in d_list:
            temp = copy.deepcopy(DOCUMENT[DOC_NAME.index(d_name)])
            doc_dict.append(temp)
            for temp_word in temp:
                if temp_word not in total_voc:
                    total_voc[temp_word] = temp[temp_word]
                else:
                    total_voc[temp_word] += temp[temp_word]

        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    return relevant_doc_word
'''