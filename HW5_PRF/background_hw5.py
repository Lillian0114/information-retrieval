import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm

snowball_stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))
doc_dict = {}
bgm = {}
# corpuslength = 0.0


def doc_tf(doc_name, list_of_words):
    # lengthofdoc = 0
    # collection = doc_name + ","
    # collection = ""
    # document = {}
    # lengthofdoc = len(list_of_words)
    for w in list_of_words: # building corpus
        w = snowball_stemmer.stem(w) 
        if w not in stop_words:
            # lengthofdoc += 1.0
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            # document[w] = document.get(w, 0.) + 1.0

    # for wc in document.keys():
    #     collection += wc+":"+str(document[wc]/lengthofdoc)+" "
    #     # collection+=wc+":"+str(document[wc])+" "

    # testfile = open("collection_docname1.txt", "a")
    # testfile.write(collection+"\n")


with open('D:/Python_test/IR/Hw5_PRF/ir_hw5_data/doc_list.txt', 'r') as L:
    for filename in tqdm(L):
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw5_PRF/ir_hw5_data/docs/"
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
with open('BGLM_hw5_unique.txt', 'w') as f:
    for key, values in bgm.items():
        text = str(key)+" "+str(values)
        f.write(str(text)+"\n")

f.close()
# '''

# BGLM_hw5_unique 152516 corpus length