from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from preprocess import Tokenizer
import sklearn.cross_validation
from sklearn.linear_model import Perceptron
import reading as rd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier






def evaluate(clf,X_train,X_test,y_train,y_test):
	clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print metrics.confusion_matrix(y_test, pred)
	print metrics.accuracy_score(y_test, pred)
	return None

vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=Tokenizer(), lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 3))
vectorizer_hash = HashingVectorizer(tokenizer=Tokenizer(), lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 3))

Xtrain, Xtest, y_train, y_test = sklearn.cross_validation.train_test_split(rd.dataset, rd.target, test_size=0.4)



#HASH
X_train = vectorizer_hash.transform(Xtrain)
X_test = vectorizer_hash.transform(Xtest)

print "SVM"
clf = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3)
evalute(clf,X_train,X_test,y_train,y_test)
print ""

print "SVM with SGD"
clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
evalute(clf,X_train,X_test,y_train,y_test)
print ""

print "Ridge Regression"
clf = RidgeClassifier(tol=1e-2, solver="lsqr")
evalute(clf,X_train,X_test,y_train,y_test)
print ""



#TFIDF
X_train = vectorizer_tfidf.fit_transform(Xtrain)
X_test = vectorizer_tfidf.fit_transform(Xtest)

print "BernoulliNB"
clf = BernoulliNB(alpha=0.01)
evalute(clf,X_train,X_test,y_train,y_test)
print ""

print "MultinomialNB"
clf = MultinomialNB(alpha=0.01)
evalute(clf,X_train,X_test,y_train,y_test)
print ""

print "Perceptron"
clf = Perceptron(n_iter=50)
evalute(clf,X_train,X_test,y_train,y_test)
print ""


print "Random Forest Classifier"
clf = RandomForestClassifier(n_estimators=10,max_features="log2")
evalute(clf,X_train,X_test,y_train,y_test)
print ""



