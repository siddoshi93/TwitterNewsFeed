from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from preprocess import Tokenizer
import sklearn.cross_validation
from sklearn.linear_model import Perceptron
import reading as rd

#   Random Forest # other

score =0.0
vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5,tokenizer=Tokenizer(), lowercase=True, strip_accents='unicode', stop_words='english', ngram_range=(1, 3))


for i in range(0,5):
    Xtrain, Xtest, y_train, y_test = sklearn.cross_validation.train_test_split(rd.dataset, rd.target, test_size=0.2)
   
    X_train = vectorizer_tfidf.fit_transform(Xtrain)
    X_test = vectorizer_tfidf.fit_transform(Xtest)

    clf = Perceptron(n_iter=50)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(metrics.confusion_matrix(y_test, pred))
    temp = metrics.accuracy_score(y_test, pred)
    score += temp
    print temp

print "Final"
print (score/5)

