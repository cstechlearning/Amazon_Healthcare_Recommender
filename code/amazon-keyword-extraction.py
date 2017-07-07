import os
os.chdir('/Users/Guang/RAKE-tutorial')
import rake
import operator
import pickle
import nltk
#nltk.download()
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import itertools
import gensim
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import pandas as pd
from pattern.text.en import singularize


'''
read file
'''

os.chdir('/Users/Guang/Downloads/')

dic ={}
for l in open('amazonproject/amazon-easy-health-50000.csv'):
    name = l.split()[0]
    content = " ".join(x for x in l.split()[1:])
    name1 = content.split(' ****** ')[0]
    content1 = content.replace(' ****** ',' ')
    name2 = name + ' ****** ' + name1
    print(name2)
    dic[name2] = content1
    

productkeys = dic.keys()
productvalues = dic.values()


print('good')
'''
Words cleaning 
'''
## capital lower and remove punctuations
tokenizer = RegexpTokenizer(r'\w+')
texts_tokenized = [[word.lower() for word in tokenizer.tokenize(x)] for x in productvalues]

print('good')
## stopwords and stemming
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stemmer.stem(word)
print('good')
texts_filtered = [[singularize(word) for word in x if not word in stop_words] for x in texts_tokenized]
texts_filtered1 = [" ".join(word for word in x) for x in texts_filtered]
print('good')

'''
rake
'''


rake_object = rake.Rake("/Users/Guang/RAKE-tutorial/SmartStoplist.txt", 3, 3, 1)

allkeywordlist = []
bigdic = {}
for i in range(len(texts_filtered1)):
    ### keyword dictionary for each product
    oneproductdic = dict(rake_object.run(texts_filtered1[i]))
    ### make bigdic for name:keyword dic 
    bigdic[productkeys[i]] = oneproductdic
    print(oneproductdic)
    ### keywordlist for ALL 
    allkeywordlist += oneproductdic.keys()
    
allkeywordset = set(allkeywordlist)  
print(len(allkeywordset))

keyworddic = {}
## kw = keyword, i=product
for kw in allkeywordset:
    print(kw)
    keyworddic[kw] = {}
    sum1 = 0
    for i in bigdic:
        if kw in bigdic[i].keys():
            keyworddic[kw][i] = bigdic[i][kw]
            sum1 += bigdic[i][kw]
    keyworddic[kw]['sum'] = sum1
    keyworddic[kw] = OrderedDict(sorted(keyworddic[kw].items(), key=lambda t:t[1], reverse=True))    
            
            
print(keyworddic["nice"])        
 
keyworddic =  OrderedDict(sorted(keyworddic.items(), key=lambda kv:kv[1]['sum'], reverse=True))
   
with open('keyword-dic-amazon-easy-health-50000.json','w') as f:
    json.dump(keyworddic,f)
f.close()
print('finish')