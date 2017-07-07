import os
os.chdir('/Users/Guang/RAKE-tutorial')
import rake
import operator
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
dicnlp = {}
for l in open('amazonproject/amazon-easy-health-50000.csv'):
    tag = l.split()[0]
    content = " ".join(x for x in l.split()[1:])
    # contentnlp = content.replace(' ****** ',' ')

    name = content.split(' ****** ')[0]
    doublename = tag + ' ****** ' + name

    print(doublename)
    dic[doublename] = content
    

pkeys = dic.keys()
pvalues = dic.values()



print('good')

# print(productvalues[0])
print("=======================================================================")

'''
Words cleaning 
'''
# ## capital lower and remove punctuations
tokenizer = RegexpTokenizer(r'\w+')
texts_tokenized = [[word.lower() for word in tokenizer.tokenize(x.replace(' ****** ',' '))] for x in pvalues]

print(texts_tokenized[0])

## stopwords and stemming
stop_words = stopwords.words('english')
texts_filtered = [[singularize(word) for word in x if not word in stop_words] for x in texts_tokenized]


print("=======================================================================")
print(texts_filtered[0])

# if need to remove those low-frequency words?

# texts_filtered1 = [" ".join(word for word in x) for x in texts_filtered]




'''
gensim
'''

#genism
dictionary = corpora.Dictionary(texts_filtered)
print(len(dictionary))
#list(itertools.islice(dictionary.token2id.items(), 0, 20))
corpus = [dictionary.doc2bow(text) for text in texts_filtered]
#print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


# numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=50000)
# s = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv=False)


lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
index = similarities.MatrixSimilarity(lsi[corpus_tfidf])   ####similarity between any two items


bigdic = {}
for i in range(len(pkeys)):
    # try:   
        print(i)
        doublename = pkeys[i]
        content = pvalues[i]
        print(content)
    
        bigdic[doublename] = {}
 
        bigdic[doublename]['name'] = content.split('****** ')[0]
        bigdic[doublename]['des'] = content.split('****** ')[1]
        bigdic[doublename]['price'] = content.split('****** ')[4]
        bigdic[doublename]['ave'] = content.split('****** ')[5]
        bigdic[doublename]['salekey'] = content.split('****** ')[7]
        # print(content.split('****** ')[7])
        bigdic[doublename]['salevalue'] = content.split('****** ')[8]
        # print(content.split('****** ')[8])

    

        l1 = lsi[dictionary.doc2bow(texts_filtered[i])]
        sort_dic = OrderedDict(sorted(enumerate(index[l1]), key=lambda item: -item[1])[0:10])
        top10dic ={}
        for i in sort_dic:
    	    top10dic[pkeys[i]] = float(sort_dic[i])
        top10dic = OrderedDict(sorted(top10dic.items(), key=lambda t:t[1], reverse=True))
        bigdic[doublename]['similar'] = top10dic
        # print(top10dic)
    # except:
    #     ValueError


with open('keyword-dic-amazon-health-50000-similarity.json','w') as f:
    json.dump(bigdic,f)
    f.close()
    
print('finish')   
# print(corpus_tfidf)
# print("======================================================================")
# print(lsi)
# print("======================================================================")
# print(index)
# for j in index:
#     print j
# print(len(index))  
# print("======================================================================")




# 3. machine learning prediction (bags of words kaggle, countverizaotr, and tfidf doesnt improve too much)
# 4. javascript
# 5. ipython (nlp yuan'li)