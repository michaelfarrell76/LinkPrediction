import pickle
import argparse
import time
import itertools
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import sys

cv_iter = 100
train_iter = 1000
clf_type = 'svm_lin'

category = 'astro-ph'
author_ind = pickle.load(open("data/" + category + '_author_ind.pkl', 'rb'))
train_adj_list = pickle.load(open("data/" + category + '_train_adj_list.pkl', 'rb'))
test_adj_list = pickle.load(open("data/" + category + '_test_adj_list.pkl', 'rb'))
pred_edges = set(test_adj_list) - set(train_adj_list)

# Build dataframes
train_df = pd.read_csv("data/" + category + "_train_df.csv")
test_df = pd.read_csv("data/" + category + "_test_df.csv")

# Build matrices
train_mat = train_df[train_df.columns[1:-1]].values
train_out = train_df[train_df.columns[-1]].values
test_mat = test_df[test_df.columns[1:-1]].values
test_out = test_df[test_df.columns[-1]].values

def build_model(clf_type, iters, C = None):
    model = None
    params = {'max_iter':iters, 'class_weight':'balanced'}
    if clf_type == 'logreg':
        model = LogisticRegression
    if clf_type == 'svm_lin':
        model = LinearSVC
    elif clf_type == 'svm_rbf':        
        model = SVC
        params['probability']= True
    if C is not None:
        params["C"] = C
    return model(**params)

def make_predictions(clf, test_mat, test_out):
    if clf_type == 'logreg' or clf_type == 'svm_rbf':
        preds = np.exp(clf.predict_log_proba(test_mat))
        probs = np.array([pred[1] for pred in preds])
    else:
        probs = clf.decision_function(test_mat)
    top_ind = probs.argsort()[-len(pred_edges):][::-1]
    acc = np.sum(test_out[top_ind]) / float(len(pred_edges))
    print 'Top-k accuracy for %s model: %0.4f' % \
        (clf_type, acc)
    return "%0.4f" % (acc)

def run(args):
	cv_iter = args.cv_iter
	train_iter = args.train_iter
	clf_type = args.clf_type

	clf = build_model(clf_type, cv_iter)
	C = [0.1, 1.0, 10.0]
	parameters = {"C": C}
	gs = GridSearchCV(clf, param_grid=parameters, cv=3, scoring="accuracy")
	gs.fit(train_mat, train_out)

	gs.best_estimator_, gs.best_params_, gs.best_score_, gs.grid_scores_
	best_clf = gs.best_estimator_
	best_accuracy = gs.best_score_
	C_opt = gs.best_params_['C']
	print "C_optimal is " + str(C_opt)

	clf = build_model(clf_type, train_iter, C=C_opt)
	clf = clf.fit(train_mat, train_out)
	acc = make_predictions(clf, test_mat, test_out)
	pickle.dump(clf, open('models/' + category + '_' + str(clf_type) + '_' + acc + '.pkl', 'wb'))


def main(arguments):
	parser = argparse.ArgumentParser(
	            description=__doc__,
	            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--clf_type', help="Type of classifier",
	                                               default='logreg')
	parser.add_argument('--cv_iter', help="Number of iterations to run cross val",
	                                               default=100)
	parser.add_argument('--train_iter', help="Number of iterations to train",
	                                               default=100)
	args = parser.parse_args(arguments)
	run(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))