import tfidf_cal
import numpy as np
from collections import defaultdict
import nltk
from tqdm import tqdm
from sklearn import preprocessing


table = tfidf_cal.TfIdf()

# bm25 parameters
K1 = 3  # doc k
# K1 = 4
b = 0.685
# b = 0.5
K2 = 0.01751
K3 = 0.2  # query k
alldoclen = []
qalldoclen=[]

i = 0

with open('D:/Python_test/IR/Hw1_VSM/ir_hw1_data/doc_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw1_VSM/ir_hw1_data/docs/"
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

print("QQ")

# '''
with open('D:/Python_test/IR/Hw1_VSM/ir_hw1_data/query_list.txt', 'r') as Q:

    for queryfilename in tqdm(Q):
        queryfilename = queryfilename.strip('\n')
        path = "D:/Python_test/IR/Hw1_VSM/ir_hw1_data/queries/"
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

with open('D:/Python_test/IR/Hw1_VSM/ir_hw1_data/query_list.txt', 'r') as Q:
    i=-1
    for queryfilename in tqdm(Q):
        i+=1
        queryfilename = queryfilename.strip('\n')
        table.BM25(K1, b, avglen, alldoclen, K3, queryfilename,K2,qalldoclen[i])
        # table.BM25(K1, b, avglen, alldoclen, K3, queryfilename)
        # table.BM25(K1,b,avglen,alldoclen,K3,queryfilename,qavglen,qalldoclen[i])

Q.close()
print("prepare to finalize!")

ans = "Query,RetrievedDocuments"
for Queryname in table.sim_ans:
    ans += "\n"+Queryname+","
    for key, value in table.sim_ans[Queryname]:
        ans += key+" "
vsm_file = open("bm25_result2.txt", "w")
vsm_file.write(ans)
vsm_file.close()
# '''


# for doc in table.sims["301"]:
#     # ans += doc+" "
#     print(doc, end=" ")

# print("\n")
# for doc in table.sims2["301"]:
# ans += doc+" "
# print(doc, end=" ")

# data = np.genfromtxt('data')
# tempalldoclen[tempalldoclen == 0.0] = np.nan
# avglen = np.nanmean(tempalldoclen[:, 1:], axis=1)
# tempalldoclen = preprocessing.normalize(tempalldoclen, norm='l2')
# tempalldoclen = [float(i)/max(tempalldoclen) for i in tempalldoclen]
# alldoclen = tempalldoclen