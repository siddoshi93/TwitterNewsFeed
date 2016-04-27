from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.cross_validation
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import reading as rd
from preprocess import Tokenizer


def get_category(x):
	ans = ''
	if x == 0:
		ans = 'politics'
	elif x==1:
		ans = 'sports'
	elif x==2:
		ans = 'technology'
	elif x==3:
		ans = 'entertainment'
	elif x==4:
		ans = 'finance'

	return ans


vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=Tokenizer(), lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 3))
clf = MultinomialNB(alpha=0.01)
train = vectorizer_tfidf.fit_transform(rd.dataset)
clf.fit(train,rd.target)

while True:
	var = raw_input('Enter tweet:\n')
	if var == 'exit':
		break
	
	temp = vectorizer_tfidf.transform([var])
	pred = clf.predict(temp)
	get_category(pred[0])

