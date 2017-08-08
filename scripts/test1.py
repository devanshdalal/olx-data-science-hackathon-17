from textblob.classifiers import NaiveBayesClassifier
import pandas as pd

train = [
	('I love this sandwich.', 'pos'),
	('This is an amazing place!', 'pos'),
	('I feel very good about these beers.', 'pos'),
	('This is my best work.', 'pos'),
	("What an awesome view", 'pos'),
	('I do not like this restaurant', 'neg'),
	('I am tired of this stuff.', 'neg'),
	("I can't deal with this", 'neg'),
	('He is my sworn enemy!', 'neg'),
	('kk hhda akjf.', 'neg2')
]
test = [
	('The beer was good.', 'pos'),
	('I do not enjoy my job', 'neg'),
	("I ain't feeling dandy today.", 'neg'),
	("I feel amazing!", 'pos'),
	('Gary is a friend of mine.', 'pos'),
	("I can't believe I'm doing this.", 'neg')
]



cl = NaiveBayesClassifier(train)

test_data = [
    ('The beer was good.'),
	('The beer was good.')
]

print( cl.classify( test_data ))  # "pos"


# print(cl.classify("Their burgers are amazing"))  # "pos"
# print(cl.classify("I don't like their pizza."))  # "neg"
# print(cl.classify('kk hhda akjf.'))
