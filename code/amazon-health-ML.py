import os
import nltk
#nltk.download()
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import itertools
import gensim
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy as sp
from pattern.text.en import singularize
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


'''
read file
'''

os.chdir('/Users/Guang/Downloads/')

dic ={}
dicnlp = {}
for l in open('amazonproject/amazon-easy-health-20000.csv'):  
    tag = l.split()[0]
    content = " ".join(x for x in l.split()[1:])
    # contentnlp = content.replace(' ****** ',' ')

    name = content.split(' ****** ')[0]
    doublename = tag + ' ****** ' + name

    print(doublename)
    dic[doublename] = content
    

pkeys = dic.keys()
pvalues = dic.values()



lprice = [[x.split('****** ')[4] for x in pvalues]]
lscore = [x.split('****** ')[5] for x in pvalues]
lsale =  [x.split('****** ')[8] for x in pvalues]

print(lprice)

lprice.append(lscore)
lprice.append(lsale)
df = pd.DataFrame(lprice).transpose()



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
texts_filtered1 = [" ".join(word for word in x) for x in texts_filtered]


print("=======================================================================")
print(texts_filtered1[0])




vectorizer = TfidfVectorizer(max_features=1000)
data = vectorizer.fit_transform(texts_filtered1)
name = vectorizer.get_feature_names()
# print(data)
df1 = pd.DataFrame(data.todense())
# df1.columns = name 
# print(df1)
print(name[999])
# print(texts_filtered1[23])

df2 = pd.concat([df1, df], axis=1, ignore_index=True)
df2.columns = name + ['Price','Score','Sales']


df2 = df2[~df2['Price'].str.contains('nan')]
df2 = df2[~df2['Sales'].str.contains('no')]
try:
    df2['Price'] = df2['Price'].astype(float)
    df2['Sales'] = df2['Sales'].astype(float)
    df2['Score'] = df2['Score'].astype(float)
except:
    ValueError

df2.to_csv('amazon-health-20000-ML.csv', sep='\t')
print('yeah')








'''
remove string in df
'''

df2['Sales'] = df2['Sales'].astype(str)
df2['Price'] = df2['Price'].astype(str)
df2['Score'] = df2['Score'].astype(str)
# df = df[df[['Sales']].apply(lambda x: x[0].isdigit(), axis=1)]
# df = df[df[['Price']].apply(lambda x: x[0].replace(".", "", 1).isdigit(), axis=1)]
df2 = df2[df2['Score'].apply(lambda x: x.strip().replace(".", "", 1).isdigit())]
df2 = df2[df2['Sales'].apply(lambda x: x.replace(".", "", 1).isdigit())]
df2 = df2[df2['Price'].apply(lambda x: x.strip().replace(".", "", 1).isdigit())]


df2['Sales'] = df2['Sales'].astype(float)
df2['Price'] = df2['Price'].astype(float)
df2['Score'] = df2['Score'].astype(float)



'''
remove string in df
'''







df3 = df2.drop(['Sales'],axis=1)


Y = df2['Sales'].values

X = df3.values     


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)



clf  =  XGBRegressor(
        learning_rate = 0.1,
        n_estimators = 200,
        max_depth = 3,
        silent = False
        )

clf.fit(X_train,Y_train)


importance = clf.feature_importances_
dfi = pd.DataFrame(importance, index=df3.columns, columns=["Importance"])
dfi = dfi.sort_values(['Importance'],ascending=False)
print(dfi)



Predicted_Train = cross_val_predict(clf, X_train, Y_train, cv=10)

#Cross-validated data Pearson's CE
Train_corr = sp.stats.pearsonr(Y_train, Predicted_Train)
print('Correlation Coefficient for Traindata is:')
print(Train_corr)


# ###MSE
MSE_Train = mean_squared_error(Y_train, Predicted_Train)
print('MSE for training data is:')
print(MSE_Train)



Predicted_Test = clf.predict(X_test)
#Test Pearson's CE
Test_corr=sp.stats.pearsonr(Y_test, Predicted_Test)
print('Correlation Coefficient for Testdata is:')
print(Test_corr)


# ###MSE
MSE_Test = mean_squared_error(Y_test, Predicted_Test)
print('MSE for test data is:')
print(MSE_Test)



First savefig,then show

Predicted_Test = clf.predict(X_test)

fig_Test, ax_Test = plt.subplots()
fig_Test.set_size_inches(15, 15)
ax_Test = sns.regplot(x=Y_test, y=Predicted_Test, scatter_kws={"color":"red",'s':60},
                  line_kws={"color":"blue","lw":3},marker="o")
plt.ylim(0, 1000000)
plt.xlim(0, 1500000)
ax_Test.set_xlabel('Real Sales Rank',fontsize=18)
ax_Test.set_ylabel('Predicted Sales Rank',fontsize=18)

plt.savefig('test.png')
plt.show()



