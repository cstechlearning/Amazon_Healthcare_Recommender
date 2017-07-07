import os
import pandas as pd 
#import pandas as pd
import numpy as np
from ast import literal_eval
import operator
import pickle
import gzip 

### run by python3

os.chdir('/Users/Guang/Downloads/')


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
    print(i)
  return pd.DataFrame.from_dict(df, orient='index')
  
df1 = getDF('amazonproject/meta_health_strict.json.gz') 
print("hahahhahahaahahahahahh")
df2 = getDF('amazonproject/reviews_Health_and_Personal_Care.json.gz')  
  
#df1 = pd.read_json('meta.json',lines=True)
#df2 = pd.read_json('reviews.json',lines=True)
#df1 = pd.read_json('meta_health_strict.json',lines=True)
# df2 = pd.read_json('reviews_Health_and_Personal_Care_5.json',lines=True)
# df2 = pd.read_json('reviews_Health_and_Personal_Care.json',lines=True)



df1['categories'] = [' '.join(str(x) for x in l[0]) for l in df1['categories']]


#df1 = df1[df1['categories'].str.contains('Joint & Muscle Pain Relief|Vitamin B|Antibiotics & Antiseptics|Medication Aids|Cold & Flu Relief')]
#df1 = df1[df1['categories'].str.contains('Cold & Flu Relief')]
df1 = df1.reset_index(drop=True)


#df1['categories'] 
# l = df1['salesRank'].iloc[2].keys()[0]
# k = df1['salesRank'].iloc[2].values()[0]
# s = str(l) + " " + str(k)
# print(s)


print("======================================")


f = open('amazon-easy-health-search-latest.csv','w')


for i in range(0,len(df1['asin'])):
    id = str(df1['asin'][i])
    des = df1['description'][i]
    category = df1['categories'][i]
    title = df1['title'][i]
    price = df1['price'][i]
    
    
    textlist = df2[(df2.ix[:, 1] == id)]['reviewText'].fillna('').values.tolist()
    sumlist = df2[(df2.ix[:, 1] == id)]['summary'].fillna('').values.tolist()
    scorelist = df2[(df2.ix[:, 1] == id)]['overall'].fillna('').values.tolist()
    votelist = df2[(df2.ix[:, 1] == id)]['helpful'].fillna(0).values.tolist()
   
    
    textstring = ' '.join(str(u) for u in textlist) 
    sumstring = ' '.join(str(u) for u in sumlist) 
    votestring = ' '.join(str(u) for u in votelist)
    
    try:
    
       if(len(scorelist)==0):
           ave = 'no' 
       else:
           ave = sum(scorelist)/float(len(scorelist))
        
    
    ####salesrank
       if(pd.isnull(df1['salesRank'][i])):
             salekey = 'no'
             salevalue = 'no'
       else:     
             salekey = list(df1['salesRank'][i])[0]
             salevalue = list(df1['salesRank'][i].values())[0]
    
    except:
        ValueError
    
    
    
    s = ( str(title) + ' ****** ' + str(des) + ' ****** ' + str(textstring) + ' ****** ' 
          + str(sumstring) + ' ****** ' + str(price) + ' ****** ' + str(ave) + ' ****** '
          + str(category) + ' ****** ' + str(salekey) + ' ****** ' + str(salevalue) )
    
    # s = str(des) + ' ******* ' + str(salekey) + ' ****** ' + str(salevalue)
    s = s.replace('\n', ' ')
   
   
    #dic[id] = s
    f.write(id + "\t" + s + "\n")

    print(i)
    
# with open('amazon-health-whole', 'wb') as f:
#     pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()  
print('finish')    



















# '''
# Sort the tags of health goods:

# dic ={}

# for i in df1['categories']:
#     for x in i[0]:
#         if x not in dic:
#             dic[x] = 1
#         else: 
#             dic[x] += 1
# print(dic)

# sorted(dic.iteritems(), key=operator.itemgetter(1))
# '''
 
    

