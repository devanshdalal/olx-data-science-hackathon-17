import pandas as pd
import numpy as np
import sys,csv
import time
import dill
import math
import operator
import random
import fasttext,os
from copy import deepcopy
from textblob.classifiers import NaiveBayesClassifier
# reload(sys)  
# sys.setdefaultencoding('utf8')

 #lambda a: a.encode('utf-8').strip()  
converter={'id':np.int64}

# tr=pd.read_csv('data/train.csv',dtype=converter).fillna(0)
# te=pd.read_csv('data/test.csv',dtype=converter).fillna(0)
# tr.description.encode('utf-8')
# tr.title.encode('utf-8')
# with open('_cache/trte.pkl', 'wb') as f:
#     dill.dump(tr, f)
#     dill.dump(te, f)
with open('_cache/trte.pkl', 'rb') as f:
	tr = dill.load(f)
	te = dill.load(f)
n=len(tr)
m=len(te)

def create_submission(values,test_data,name):
	d = {'id':test_data.id, 'title':test_data.title, 'description':test_data.description ,'category':pd.Series(values)}
	submission= pd.DataFrame(d)
	submission.to_csv(name)

a=set(tr.category)
hash={}
rev_hash={}
idx=0
for x in a:
	hash[x]=idx
	rev_hash[idx]=x
	idx+=1
print(a)
print(hash)
print(rev_hash)

def fasttest(train_data,test_data):
	td=[]
	os.system('rm -rf data.train.txt')
	with open ('data.train.txt', 'w') as f: 
		for i,row in train_data.iterrows():
			if(i%10000==0):
				print('t',i)
			desc,title,cat=row['description'],row['title'],row['category']
			# print(desc,title,cat)
			f.write( '__label__' + str(hash[cat]) + ' ' + str(title) +'\n' )
	classifier = fasttext.supervised('data.train.txt', 'model')
	testd=[]
	for i,row in test_data.iterrows():
		desc,title=row['description'],row['title']
		testd.append(title)
	values=classifier.predict(testd)
	# print(values)
	values=list(map(lambda x:str(rev_hash[int(x[0])]),values))
	# print(values)
	create_submission(np.array(values),test_data,'f.csv')

fraction=0.9
# nbcf(tr.ix[:fraction*n],tr.ix[n*fraction:],True)
# fasttest(tr.ix[:100],te.ix[100:110])
fasttest(tr,te)





		