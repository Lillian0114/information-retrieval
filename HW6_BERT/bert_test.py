import pandas as pd
import random
import numpy as np

document_csv_path = './HW6_dataset/documents.csv'
training_csv_path = './HW6_dataset/train_queries.csv'
testing_csv_path = './HW6_dataset/test_queries.csv'
# Input limitation
max_query_length = 64
max_input_length = 512
num_negatives = 3   # num. of negative documents to pair with a positive document

dev_set_ratio = 0.2


def preprocess_df(df):
    ''' Preprocess DataFrame into training instances for BERT. '''
    instances = []
    j = 0
    # Parse CSV
    for i, row in df.iterrows():
        query_id, query_text, pos_doc_ids, bm25_top1000, _ = row
        pos_doc_id_list = pos_doc_ids.split()
        pos_doc_id_set = set(pos_doc_id_list)
        bm25_top1000_list = bm25_top1000.split()
        bm25_top1000_set = set(bm25_top1000_list)

        # Pair BM25 neg. with pos. samples
        labeled_pos_neg_list = []
        for pos_doc_id in pos_doc_id_list:
            neg_doc_id_set = bm25_top1000_set - pos_doc_id_set
            neg_doc_ids = random.sample(neg_doc_id_set, num_negatives)
            pos_position = random.randint(0, num_negatives)
            pos_neg_doc_ids = neg_doc_ids
            pos_neg_doc_ids.insert(pos_position, pos_doc_id)
            labeled_sample = (pos_neg_doc_ids, pos_position) #
            labeled_pos_neg_list.append(labeled_sample)
            if j==2:
                # print(labeled_sample)
                print(labeled_pos_neg_list)
                break
            else:
                j+=1

train_df = pd.read_csv(training_csv_path)
dev_df, train_df = np.split(train_df, [int(dev_set_ratio*len(train_df))])
dev_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)
preprocess_df(train_df)
# train_instances = preprocess_df(train_df)


'''
doc_id_to_text = {}
doc_df = pd.read_csv(document_csv_path)
# print(doc_df)
doc_df.fillna("<Empty Document>", inplace=True)
id_text_pair = zip(doc_df["doc_id"], doc_df["doc_text"])
for i, pair in enumerate(id_text_pair, start=1):
    doc_id, doc_text = pair
    doc_id_to_text[doc_id] = doc_text
    
    print("Progress: %d/%d\r" % (i, len(doc_df)), end='')
    # if i==2:
        # break
    
# print(doc_df.tail())
# print(doc_id_to_text)
'''