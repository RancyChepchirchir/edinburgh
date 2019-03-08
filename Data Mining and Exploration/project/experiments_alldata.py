###
# run a few experiments
###
# INPUT:
#	fulldata from join_data.py
#	keys_content_features: array(String) of column keys
# PRINTS:
#	training accuracy for Dummy, GaussianNB, LogisticRegression and MLPClassifier
###

from scripts import *

(X_content_full, keys_content_features) = read_transform_contentfeatures()
(y_full,keys_labels) = read_transform_labels()
fulldata = join_data(X_content_full, y_full)

X = fulldata[keys_content_features]
y = fulldata['Neutrality']

###

from sklearn.dummy import DummyClassifier

dc = DummyClassifier(strategy='prior').fit(X,y)
print( 'Dummy accuracy: {}'.format( dc.score(X,y) ) )

###

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X,y)
print( 'GaussianNB accuracy: {}'.format( gnb.score(X,y) ) )

###

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(X,y)
print( 'LogisticRegression accuracy: {}'.format( logreg.score(X,y) ) )

###

from sklearn.neural_network import MLPClassifier

nnet = MLPClassifier(hidden_layer_sizes=(100,50),activation='logistic').fit(X,y)
print( 'MLPClassifier accuracy: {}'.format( nnet.score(X,y) ) )