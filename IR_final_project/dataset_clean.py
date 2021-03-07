import re 
import csv
import pandas as pd
import json

def delteurl(delete_string): 
    if "\n" in delete_string: 
        delete_string = delete_string.replace('\n',' ')
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    # regex = r'^https?:\/\/.*[\r\n]*'
    urlold = re.findall(regex,delete_string)
    if len(urlold)!=1:
        url=([x[0] for x in urlold])
    else:
        url = "".join( [x[0] for x in urlold] ) 

    if type(url) is str:
        if len(url)>1:
            delete_string = delete_string.replace(url, '')
            delete_string = "".join(delete_string)
    else: 
        for i in range(len(url)):
            if len(url)>1:
                delete_string = delete_string.replace(url[i], '')
                delete_string = "".join(delete_string)
    if "@" in delete_string:
        regex = r'@[\w]*'
        people = re.findall(regex,delete_string)  
        if len(people)>=1:
            for i in range(len(people)):
                delete_string = delete_string.replace(people[i],'')
        else:
            delete_string = delete_string.replace(people,'')
        delete_string = delete_string.replace('RT : ','')
    
    return delete_string

textlist = []
classlist = []
# classid = ['Politics_temp']
classid = ['Politics','Education','Health','Marketing','Music','News','Sport','Technology','Pets','Food','Family']
i=0
for i in range(len(classid)):
    if classid[i]=="Marketing":
        textname = 'D:/Python_test/IR/Data_set/'+classid[i]+"_new"+'.txt'
    else:
        textname = 'D:/Python_test/IR/Data_set/'+classid[i]+'.txt'
    with open(textname,encoding="utf-8", newline='') as f:
        rows = f.readlines()
        for row in rows:
            twitter_dict = json.loads(row)
            # print(twitter_dict)
            if 'retweeted_status' in twitter_dict:
                if 'extended_tweet' in twitter_dict['retweeted_status']:
                    fulltext=delteurl(twitter_dict['retweeted_status']['extended_tweet']['full_text'])
                else:
                    fulltext = delteurl(twitter_dict['text'])
            elif twitter_dict['truncated']=='True':
                fulltext=delteurl(twitter_dict['extended_tweet']['full_text'])
            else:
                fulltext = delteurl(twitter_dict['text'])

            textlist.append(fulltext)
            classlist.append(classid[i])

# '''
listlen = len(textlist)
# print(listlen)
dict_save = {
    "id": pd.Series(range(listlen)), 
    "text": textlist,
    "class" : classlist
    # "createtime": timelist,
    # "user": useridlist
    }
df = pd.DataFrame(dict_save)

df.to_csv ('D:/Python_test/IR/Final_represent/tweet_data_final.txt', encoding='utf-8-sig',mode='a', index = False, header=True)