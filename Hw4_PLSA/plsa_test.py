from datetime import *
import math
import json
import numpy as np
import pandas as pd

import gc
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))

K = 10
MaxIter = 100
Stop_Threshold = 10

Param_Alpha = 0.45
Param_Beta = 0.35


def GetWords(QueryList, DocList):
    ID2Word = {}
    Word2ID = {}
    CurrentID = 0
    
    for doc in DocList.values():
        for line in open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/docs/"+doc+".txt","r").readlines():
            for word in line.split()[:-1]:
                word = snowball_stemmer.stem(word)
                if word not in Word2ID.keys():
                    if word not in stop_words:
                        ID2Word.update({CurrentID : word})
                        Word2ID.update({word : CurrentID})
                        CurrentID += 1
    
    for query in QueryList.values():
        for line in open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/queries/"+query+".txt","r").readlines():
            for word in line.split()[:-1]:
                word = snowball_stemmer.stem(word)
                if word not in Word2ID.keys():
                    if word not in stop_words:
                        ID2Word.update({CurrentID : word})
                        Word2ID.update({word : CurrentID})
                        CurrentID += 1

    return ID2Word, Word2ID

def Preprocessing(DocList, ID2Word):
    N = len(DocList)
    M = len(ID2Word)
    del ID2Word
    A = np.zeros([N, M], int)

    for index, doc in enumerate(DocList.values()):
        for line in open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/docs/"+doc+".txt","r").readlines():
            for word in line.split()[:-1]:
                word = snowball_stemmer.stem(word)
                if word not in stop_words:
                    A[index,Word2ID[word]] += 1
                    
    del DocList
    
    return N, M, A

def InitParam(M,N,K):
    LAMBDA = np.random.random([N,K])
    THETA = np.random.random([K,M])
    for i in range(N):
        LAMBDA[i,] /= np.sum(LAMBDA[i,])
    for i in range(K):
        THETA[i,] /= np.sum(THETA[i,])
            
    return LAMBDA, THETA

def EStep(P,M,N,K):
    for i in range(N):
        for j in range(M):
            for k in range(K):
                P[i,j,k] = THETA[k,j] * LAMBDA[i,k]
            s = np.sum(P[i,j,:])
            if s == 0:
                for k in range(K):
                    P[i,j,k] = 0
            else:
                for k in range(K):
                    P[i,j,k] /= s
                    
    return P

def MStep(A,P,LAMBDA,Theta,M,N,K):
    t = datetime.now()
    for k in range(K):
        for j in range(M):
            THETA[k,j] = np.sum(A[:,j] * P[:,j,k])
        s = np.sum(THETA[k,:])
        if s == 0:
            for j in range(M):
                THETA[k,j] = 1.0 / M
        else:
            for j in range(M):
                THETA[k,j] /= s
    print(datetime.now()-t)
    
    for i in range(N):
        for k in range(K):
            LAMBDA[i,k] = np.sum(A[i,:] * P[i,:,k])
            s = np.sum(A[i,:])
            if s == 0:
                LAMBDA[i,k] = 1.0 / K
            else:
                LAMBDA[i,k] /= s
                
    return LAMBDA, THETA

def CurrentLogLikelihood(A,LAMBDA,THETA,M,N,K):
    LogLikelihood = 0
    for i in range(N):
        for j in range(M):
            tmp = 0
            for k in range(K):
                tmp += THETA[k,j] * LAMBDA[i,k]
            if tmp > 0:
                LogLikelihood += A[i,j] * math.log(tmp)

    return LogLikelihood

QueryList = {index : queries.strip('\n') for index, queries in enumerate(open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/query_list.txt","r"))}

DocList = {index : docs.strip('\n') for index, docs in enumerate(open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/doc_list.txt","r"))}

BGLM = {index:float(lines.split()[1]) for index,lines in enumerate(open("D:/Python_test/IR/BGLM_new.txt","r"))}

ID2Word, Word2ID = GetWords(QueryList, DocList)

N, M, A = Preprocessing(DocList, ID2Word)

# LAMBDA[i,k] = p(Tk|Di)
# THETA[i,j] = p(Wj|Ti)
LAMBDA,THETA = InitParam(M,N,K)

# P[i,j,k] = p(Tk|Di,Wj)
P = np.zeros([N,M,K])

OldLogLikelihood = 1
NewLogLikelihood = 1
for i in range(MaxIter):
    t = datetime.now()
    P = EStep(P,M,N,K)
    LAMBDA, THETA = MStep(A,P,LAMBDA,THETA,M,N,K)
    NewLogLikelihood = CurrentLogLikelihood(A,LAMBDA,THETA,M,N,K)
    if (OldLogLikelihood != 1) and (NewLogLikelihood - OldLogLikelihood) < Stop_Threshold:
        break
    print(str(i) + " " + str(NewLogLikelihood) + " " + str(NewLogLikelihood - OldLogLikelihood) + " " + str(datetime.now() - t))
    OldLogLikelihood = NewLogLikelihood

A_Normalize = np.zeros([N,M],float)

for i in range(N):
    A_Normalize[i,] = np.divide(A[i,],np.sum(A[i,]))

f = open("submission.txt", "w")
f.write("Query,RetrievedDocuments\r\n")

for index, query in QueryList.items():
    f.write(query + ",")
    Score = {}
    for i,doc in DocList.items():
        s = 0
        for line in open("D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/queries/"+query+".txt","r").readlines():
            for word in line.split()[:-1]:
                a1 = np.log(Param_Alpha) + np.log(A_Normalize[i,Word2ID[word]])
                a2 = np.log(np.sum(LAMBDA[i,:] * THETA[:,Word2ID[word]])) + np.log(Param_Beta)
                a3 = np.log(1 - Param_Alpha - Param_Beta) + BGLM[Word2ID[word]]
                s += np.logaddexp(np.logaddexp(a1,a2),a3)
        Score.update({doc : s})
    Score_Sort = sorted(Score.items(), key=lambda Score: Score[1],reverse=True)
    
    i=0
    for item in Score_Sort:
        f.write(item[0] + " ")
        if i==999:
            break
        else:
            i+=1
    f.write("\r\n")
f.close()