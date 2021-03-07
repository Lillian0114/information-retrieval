import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import math

snowball_stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))
doc_dict = {}
docidf = {}  # doc idf
bgm = {}
# corpuslength = 0.0


def doc_tf(doc_name, list_of_words):
    lengthofdoc = 0
    collection = doc_name + ","
    # collection = ""
    document = {}
    # building corpus
    # lengthofdoc = len(list_of_words)
    for w in list_of_words:
        w = snowball_stemmer.stem(w)  # 詞性還原
        if w not in stop_words:
            lengthofdoc += 1.0
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            document[w] = document.get(w, 0.) + 1.0

    for wc in document.keys():
        collection+=wc+":"+str(document[wc]/lengthofdoc)+" "
        # collection+=wc+":"+str(document[wc])+" "

    testfile =  open("collection_docname.txt", "a")
    testfile.write(collection+"\n")


with open('D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/doc_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw4_PLSA/ir_hw4_data/docs/"
        file = path+filename+".txt"

        with open(file, 'r') as f:
            listOfWords = []
            for lines in f.readlines():
                listOfWords = nltk.word_tokenize(lines)
                f.close()

            doc_tf(filename, listOfWords)

L.close()

# '''
# print(len(doc_dict.keys()))
for w in doc_dict.keys():
    # bgm[w] = doc_dict[w]/sum(doc_dict.values())
    bgm[w] = doc_dict[w]/len(doc_dict.keys())
with open('BGLM.txt', 'w') as f:
    for key, values in bgm.items():
        text = str(key)+" "+str(values)
        f.write(str(text)+"\n")

f.close()
# '''


'''
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
'''
