from sys import stdin
import math

docmodel = {} #store the doc you get from ur IR model
relevant = {} #store the relevant doc 'the correct ans'

querycount = int(stdin.readline().strip())
for i in range(querycount):
    docfind = stdin.readline()
    relans = stdin.readline()
    docmodel['query'+str(i)] = docfind.strip().split(" ")
    relevant['query'+str(i)] = relans.strip().split(" ")

average_precision = 0.0
for key in relevant.keys():
    precision = 0.0
    index = 0.0
    found = 0.0
    for doc in docmodel[key]:
        index +=1.0
        if doc in relevant[key]:
            found +=1.0
            precision += found/index
    average_precision += precision/len(relevant[key])
MAP= average_precision/float(querycount)
print( float(format(MAP, '0.4f')) )

# if str(querycount).isdigit() and querycount!=0:
    