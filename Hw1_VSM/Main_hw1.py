import tfidf
import nltk
table = tfidf.TfIdf()
# i=0

#Read doc and build corpus
with open('D:/Python_test/IR/Hw1_VSM/ir_hw1_data/doc_list.txt', 'r') as L:
    for filename in L:
        filename = filename.strip('\n')
        path = "D:/Python_test/IR/Hw1_VSM/ir_hw1_data/docs/"
        file = path+filename+".txt"
        # i+=1
        with open(file, 'r') as f:
            listOfWords = []
            for lines in f.readlines():
                listOfWords += nltk.word_tokenize(lines)
                f.close()
            table.add_document(filename, listOfWords)
        # if(i==2):
        #     break
    # print(table.docidf)

L.close()


#Read query and calculate the similarity
with open('D:/Python_test/IR/Hw1_VSM/ir_hw1_data/query_list.txt', 'r') as Q:

    for queryfilename in Q:
        queryfilename = queryfilename.strip('\n')
        path = "D:/Python_test/IR/Hw1_VSM/ir_hw1_data/queries/"
        file = path+queryfilename+".txt"
        with open(file, 'r') as f:
            listOfWords = []
            for line in f.readlines():
                listOfWords += nltk.word_tokenize(line)
                f.close()

        table.similarities(queryfilename, listOfWords)
        # break

print("prepare to finalize!")

# build the answer doc
ans = "Query,RetrievedDocuments"
print(ans,end="")
for Queryname in table.sims2:
    ans += "\n"+Queryname+","
    for key,value in table.sims2[Queryname]:
        ans += key+" "
        # print(doc, end=" ")

vsm_file = open("vsm_result_final.txt", "w")
vsm_file.write(ans)
vsm_file.close()
