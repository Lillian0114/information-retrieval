import bm25_tfidf_cal
import numpy as np
from collections import defaultdict
import nltk
from tqdm import tqdm
from sklearn import preprocessing

table = bm25_tfidf_cal.TfIdf()

# bm25 parameters
K3 = 0.2  # query k
# K1 = 3  # doc k
K1 = 0.8
# b = 0.7
b = 0.7
# K2 = 0.03
K2 = 0.07
# delta = 0.4
delta = 0.2

alldoclen = []
qalldoclen=[]

i = 0

with open('D:/Python_test/IR/Hw5_PRF/ir_hw5_data/doc_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw5_PRF/ir_hw5_data/docs/"
        file = path+filename+".txt"
        i += 1
        with open(file, 'r') as f:
            listOfWords = []
            for lines in f.readlines():
                listOfWords = nltk.word_tokenize(lines)
                f.close()

            alldoclen.append(len(listOfWords))
            table.doc_tf(filename, listOfWords)
        # if(i == 3634):
            # print(listOfWords)
            # break
    # print(table.documents)

L.close()
tempalldoclen = np.array(alldoclen)
avglen = np.mean(tempalldoclen[np.nonzero(tempalldoclen)])

# table.determine_tf()

print("QQ")

# '''
with open('D:/Python_test/IR/Hw5_PRF/ir_hw5_data/query_list.txt', 'r') as Q:

    for queryfilename in tqdm(Q):
        queryfilename = queryfilename.strip('\n')
        path = "D:/Python_test/IR/Hw5_PRF/ir_hw5_data/queries/"
        file = path+queryfilename+".txt"
        with open(file, 'r') as f:

            listOfWords = []
            temp = []
            for line in f.readlines():
                listOfWords = nltk.word_tokenize(line)
                f.close()
            qalldoclen.append(len(listOfWords))
            table.query_tf(queryfilename, listOfWords)
    # break

Q.close()
# print(len(table.queidf))

tempalldoclen = np.array(qalldoclen)
qavglen = np.mean(tempalldoclen)

with open('D:/Python_test/IR/Hw5_PRF/ir_hw5_data/query_list.txt', 'r') as Q:
    i=-1
    for queryfilename in tqdm(Q):
        i+=1
        queryfilename = queryfilename.strip('\n')
        table.BM25(K1, b, avglen, alldoclen, K3, queryfilename,K2,qalldoclen[i],delta)
        # table.BM25(K1, b, avglen, alldoclen, K3, queryfilename)
        # table.BM25(K1,b,avglen,alldoclen,K3,queryfilename,qavglen,qalldoclen[i])

Q.close()
print("prepare to finalize!")

ans = "Query,RetrievedDocuments"
for Queryname in table.sim_ans:
    i=0
    ans += "\n"+Queryname+","
    for key, value in table.sim_ans[Queryname]:
        ans += key+" "
        if i==9:
            break
        else:
            i+=1
ans+="\n"
vsm_file = open("bm25_relevant_10_0.03.txt", "w")
vsm_file.write(ans)
vsm_file.close()
# '''