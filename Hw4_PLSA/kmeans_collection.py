import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import math

snowball_stemmer = SnowballStemmer('english')

documents=[]
filenames = []
stop_words = set(stopwords.words('english'))
doctf = {}  # 每篇doc的tf

def doc_tf(doc_name, list_of_words):
    # building a dictionary
    doc_dict = {}
    for w in list_of_words:
        w = snowball_stemmer.stem(w)  # 詞性還原
        if w not in stop_words:
            # TF
            doc_dict[w] = (doc_dict.get(w, 0.) + 1.0)

    doctf[doc_name] = doc_dict


with open('D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/doc_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/docs/"
        file = path+filename+".txt"
        filenames.append(filename)
        with open(file, 'r') as f:
            listOfWords = ""
            for lines in f.readlines():
                listOfWords +=lines
                f.close()

            # doc_tf(filename, listOfWords)
            documents.append(listOfWords)

L.close()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)



# i=0
# collect_kmeans={}
# for doc in documents:
#     # print(np.array(doc))
#     Y = vectorizer.transform(doc.split('\x00'))
#     prediction = model.predict(Y)
#     # print(prediction[0])
#     tempstring=collect_kmeans.get('topic'+str(prediction[0]), '')
#     collect_kmeans['topic'+str(prediction[0])]=filenames[i]+","+tempstring
#     i+=1
# # print(collect_kmeans)

# for key,topic in collect_kmeans.items():
#     docset = topic.split(',')
#     # print(topic)
#     # print(docset)
#     newdoc = ""
#     for filename in docset:
#         if filename == "" :
#             continue
#         path = "D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/docs/"
#         file = path+filename+".txt"
#         with open(file, 'r') as f:
#             listOfWords = ""
#             for lines in f.readlines():
#                 listOfWords +=lines
#                 # f.close()
#             newdoc+=listOfWords+" "
#     # doc_tf(key,newdoc)

#     testfile =  open("Collection_file10.txt", "a")
#     testfile.write(newdoc+"\n")
    

# kmeanswordcountperdoc = []
# with open('D:/Python_test/IR/Collection_file.txt', 'r') as L:
#     for lines in tqdm(L):
#         lines = lines.strip('\n')
#         # print(lines)
#         listOfWords = []
#         listOfWords = nltk.word_tokenize(lines)
#         print(listOfWords)
#         wordCountCurrentDoc = {}
#         for w in listOfWords:
#             w = snowball_stemmer.stem(w) 
#             if w in stop_words:
#                 continue
#             else :
#                 wordCountCurrentDoc[w] = wordCountCurrentDoc.get(w, 0.) + 1.0 #tf
#         print(wordCountCurrentDoc)
#         kmeanswordcountperdoc.append(wordCountCurrentDoc)
#         break

# print(kmeanswordcountperdoc)

