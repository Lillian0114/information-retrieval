import numpy as np
import os
import re
import time
import math

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm

snowball_stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))

# wordCount = {}  # it's idf, (key : word , value : word 在全部doc中出現的次數)
# it's tf(list save dict : every key is word，value is word count in each doc)
wordCountPerQuery = []
BG = {}
TFperDoc = []
sims = {}
docfilename = []
queryfilename = []

alpha = 0.988
beta = 0.012


def ReadPreprocessFile():
    # i = 0
    # with open('D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/doc_list.txt') as doc_file:
    #     for filename in doc_file:
    #         filename = filename.strip('\n')
    #         docfilename.append(filename)

    with open('D:/Python_test/IR/BGLM_new.txt') as BG_file:
        for val_voc in BG_file:
            bg_split = re.split(' ', val_voc)
            BG[bg_split[0]] = float(bg_split[1])
            # break

    with open('D:/Python_test/IR/collection_docname.txt') as wordtf:
        for doc_tf in wordtf:
            # i += 1
            temptfdic = {}
            doc_tf = re.split(' ', doc_tf)
            doc_tf = list(filter('\n'.__ne__, doc_tf))
            for pertf in doc_tf:
                tf_split = re.split(':|,', pertf)
                if len(tf_split) == 3:
                    docfilename.append(tf_split[0])
                    temptfdic[tf_split[1]] = float(tf_split[2])
                else:
                    temptfdic[tf_split[0]] = float(tf_split[1])
            TFperDoc.append(temptfdic)
            # if i == 2:
            #     break


ReadPreprocessFile()
# print(TFperDoc)

# i = 0
with open('D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/query_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/queries/"
        file = path+filename+".txt"
        queryfilename.append(filename)
        with open(file, 'r') as f:
            listOfWords = []
            # i += 1
            for lines in f.readlines():
                listOfWords = nltk.word_tokenize(lines)
                list_of_words = []
                for w in listOfWords:
                    w = snowball_stemmer.stem(w)  # 詞性還原
                    if w not in stop_words:
                        list_of_words.append(w)
            wordCountPerQuery.append(list_of_words)
            f.close()
            # querydictcount(filename, listOfWords)
            # if i == 3:
            # break

# print(wordCountPerQuery)

for querindex, querdict in tqdm(enumerate(wordCountPerQuery)):
    # print(quer)
    first = 0.0
    third = 0.0
    scoreDic = {}
    for docindex, docdict in enumerate(TFperDoc):
        # print(doc.keys())
        # print(doc) #每個word的每個字
        # finalscore = 0.0
        finalscore = 1.0
        for queryword in querdict:
            # first = alpha+np.log10(docdict.get(queryword,0.))
            first = alpha*docdict.get(queryword,0.)
            third = beta*BG.get(queryword)
            finalscore *= (third+first)

        scoreDic[docfilename[docindex]] = np.log(finalscore)

    sims[queryfilename[querindex]] = sorted(
        scoreDic.items(), key=lambda d: d[1], reverse=True)
    # break


# '''
ans = "Query,RetrievedDocuments"
for Queryname in sims:
    i = 0
    ans += "\n"+Queryname+","
    for key, value in sims[Queryname]:
        ans += key+" "
        if i == 999:
            break
        i += 1

plsa_file = open("plsa_test2.txt", "w")
plsa_file.write(ans)
plsa_file.close()
# '''
