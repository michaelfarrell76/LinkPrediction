import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from utils.build_df import build_df

category = 'astro-ph'
author_ind = pickle.load(open("data/" + category + '_author_ind.pkl', 'rb'))
train_adj_list = pickle.load(open("data/" + category + '_train_adj_list.pkl', 'rb'))
test_adj_list = pickle.load(open("data/" + category + '_test_adj_list.pkl', 'rb'))
pred_edges = set(test_adj_list) - set(train_adj_list)

# Build dataframes
train_df = build_df(train=True)
test_df = build_df(train=False)

# Build matrices
num_trees = 500
train_mat = train_df[train_df.columns[1:-1]].values
train_out = train_df[train_df.columns[-1]].values
test_mat = test_df[test_df.columns[1:-1]].values
test_out = test_df[test_df.columns[-1]].values

print 'Training Decision Tree Models'
params = {"objective": "binary:logistic",
          #"eta": eta,
          #"max_depth": max_depth,
          #"min_child_weight": min_child_weight,
          "silent": 1,
          #"seed": 1,
          #"lambda":lambda_p,
          #"alpha":alpha_p,
          "eval_metric": "error"}

gbm = xgb.train(params, xgb.DMatrix(train_mat, train_out), num_trees)
probs = gbm.predict(xgb.DMatrix(test_mat))
top_ind = probs.argsort()[-len(pred_edges):][::-1]
acc = np.sum(test_out[top_ind]) / float(len(pred_edges))
print 'Top-k accuracy for decision trees: %0.4f' % \
    (acc)