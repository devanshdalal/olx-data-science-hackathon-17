import pandas as pd
import numpy as np
import sys,csv
import time
import dill
import math
import operator
import codecs
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
	submission = test_data.assign(category=pd.Series(values))
	submission.to_csv('submissions/'+name)

def nbcf(train_data,test_data,check=False):
	td=[]
	for i,row in train_data.iterrows():
		if(i%10000==0):
				print('t',i)
		desc,title,cat=row['description'],row['title'],row['category']
		# print(desc,title,cat)
		td.append(( str(title) + " " + str(desc) ,cat))
	print('training model')
	cl = NaiveBayesClassifier(td)
	print('testing model')
	if check:
		testd=[]
		for i,row in test_data.iterrows():
			if(i%10000==0):
				print('i',i)
			desc,title,cat=row['description'],row['title'],row['category']
			testd.append(( str(title) + " " + str(desc) ,cat))
		print('accu',cl.accuracy(testd))
	else:
		values=[]
		for i,row in test_data.iterrows():
			desc,title=row['description'],row['title']
			values.append( cl.classify(( str(title) + " " + str(desc) ,'utf8') ) )
		create_submission(values,test_data,'s.csv')

fraction=0.9
nbcf(tr.ix[:fraction*n],tr.ix[n*fraction:],True)
# nbcf(tr.ix[:100],tr.ix[100:200],True)





		