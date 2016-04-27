from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.cross_validation
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import reading as rd
from preprocess import Tokenizer


score1 =0.0
score2 =0.0

vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=Tokenizer(), lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 3))


for i in range(0,5):
	Xtrain, Xtest, y_train, y_test = sklearn.cross_validation.train_test_split(rd.dataset, rd.target, test_size=0.2)


	X_train = vectorizer_tfidf.fit_transform(Xtrain)
	X_test = vectorizer_tfidf.transform(Xtest)


	clf = MultinomialNB(alpha=0.01)

	
	clf.fit(X_train, y_train)

	pred = clf.predict(X_test)
	print(metrics.confusion_matrix(y_test, pred))

	temp1 = metrics.accuracy_score(y_test, pred)
	score1 += temp1
	print temp1
	
	clf = BernoulliNB(alpha=0.01)

	clf.fit(X_train, y_train)

	pred = clf.predict(X_test)
	print(metrics.confusion_matrix(y_test, pred))

	temp2 = metrics.accuracy_score(y_test, pred)
	score2 += temp2
	print temp2
	

print "Final"
print (score1/10)
print (score2/10)