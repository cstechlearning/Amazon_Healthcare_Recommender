import os
import pandas as pd 
import numpy as np
import operator
from ast import literal_eval
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



### Create dataframe for both meta-data and reviews  
df1 = getDF('amazonproject/meta_health_strict.json.gz') 

df2 = getDF('amazonproject/reviews_Health_and_Personal_Care.json.gz')  




df1['categories'] = [' '.join(str(x) for x in l[0]) for l in df1['categories']]
df1 = df1.reset_index(drop=True)





### Obtain ReviewText, ReviewSummary from df2.
def getfromdf2(item):
   list1 = df2[(df2.ix[:, 1] == idx)]['reviewText'].fillna('').values.tolist()
   str1 = ' '.join(str(u) for u in list1)
   return str1;


f = open('amazon-easy-health.csv','w')


for i in range(0,len(df1['asin'])):
    
    idx = str(df1['asin'][i])
    des = df1['description'][i]
    category = df1['categories'][i]
    title = df1['title'][i]
    price = df1['price'][i]
    
    
   
    
    textstring = getfromdf2('reviewText') 
    sumstring =  getfromdf2('summary')
    votestring = getfromdf2('helpful')
    
 

    ### Review score calculation
    scorelist = df2[(df2.ix[:, 1] == idx)]['overall'].fillna('').values.tolist()
    if(len(scorelist)==0):
           ave = 'no' 
    else:
           ave = sum(scorelist)/float(len(scorelist))
        
    


    ####Sales Rank
    if(pd.isnull(df1['salesRank'][i])):
             salekey = 'no'
             salevalue = 'no'
    else:     
             salekey = list(df1['salesRank'][i])[0]
             salevalue = list(df1['salesRank'][i].values())[0]
    
  
    
    
    
    s = ( str(title) + ' ****** ' + str(des) + ' ****** ' + str(textstring) + ' ****** ' 
          + str(sumstring) + ' ****** ' + str(price) + ' ****** ' + str(ave) + ' ****** '
          + str(category) + ' ****** ' + str(salekey) + ' ****** ' + str(salevalue) )
    
    
    s = s.replace('\n', ' ')
   
   
    
    f.write(id + "\t" + s + "\n")
    

f.close()  
  





