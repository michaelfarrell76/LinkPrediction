import pickle
import time
import itertools
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd

def gen_neighbors(adj_list, author_ind):
    neighbors = {}
    for a in author_ind.iterkeys():
        neighbors[author_ind[a]] = frozenset()
    for e in adj_list:
        neighbors[e[0]] = neighbors[e[0]].union(set([e[1]]))
        neighbors[e[1]] = neighbors[e[1]].union(set([e[0]]))
    return neighbors

def build_year_dicts(adj_list):
    edge_year_dict = {}
    for e in adj_list:
        edge_year_dict[(e[0], e[1])] = e[2]
    return edge_year_dict

def common_neighbors(pos_edges, neighbors, authors):
    ranking = []
    for a1, a2 in pos_edges:
        neighbor_count = len(neighbors[a1].intersection(neighbors[a2]))
        ranking.append((a1, a2, neighbor_count))
    return ranking

def adamic_adar(pos_edges, neighbors):
    ranking = []
    for e in tqdm(pos_edges):
        edge_neighbors = neighbors[e[0]].intersection(neighbors[e[1]])
        score = np.sum([(1.0/np.log(len(neighbors[n]) + 1e-10)) for n in edge_neighbors])
        ranking.append((e[0], e[1], score))
    return ranking

def build_matrix(adj_list, author_ind):
    P = np.zeros((len(author_ind), len(author_ind)))
    for e in adj_list:
        P[e[0]][e[1]] = 1
        P[e[1]][e[0]] = 1        
    return P.astype(int)

# Returns a vector of the number of neighbors for 
# each node that are d away.
def gen_neighbor_count(d, dist, num_authors):
    counts = np.zeros(num_authors)
    for i in range(num_authors):
        for j in range(i+1, num_authors):
            if dist[i][j] == d:
                counts[i] += 1
                counts[j] += 1
    return counts        
        
    
# Transforms a list of results of neighbors into a matrix
# where matrix values correspond to that feature
def transform_matrix(edges, num_authors):
    m = np.zeros((num_authors, num_authors))
    for e in edges:
        m[e[0]][e[1]] = e[2]
        m[e[1]][e[0]] = e[2]
    return m

# Given matrix that is only filled triangularly for j > i, 
# fill it also for i > j.
def fill_matrix(m, num_authors):
    for i in range(num_authors):
        for j in range(i+1, num_authors):
            m[j][i] = m[i][j]
    return m

def build_df(train=True, write=False, sample=0.999):
    category = 'astro-ph'
    loc = 'data/' + category
    entries = pickle.load(open(loc + '_entries.pkl', 'rb'))
    author_ind = pickle.load(open(loc + '_author_ind.pkl', 'rb'))
    train_adj_list = pickle.load(open(loc + '_train_adj_list.pkl', 'rb'))
    test_adj_list = pickle.load(open(loc + '_test_adj_list.pkl', 'rb'))
    
    train_adj_list_with_year = pickle.load(open(loc + '_train_adj_list_with_year.pkl', 'rb'))
    test_adj_list_with_year = pickle.load(open(loc + '_test_adj_list_with_year.pkl', 'rb'))

    num_authors = len(author_ind)
    authors = range(num_authors)
    all_edges = set([(min(a1, a2), max(a1, a2)) for (a1, a2) in \
                     itertools.combinations(authors, 2)])
    pos_edges = all_edges - set(train_adj_list)
    pred_edges = set(test_adj_list) - set(train_adj_list)
    dist = pickle.load(open(loc + '_dist.pkl', 'rb'))

    neighbors = gen_neighbors(train_adj_list, author_ind)
    cn_edges = common_neighbors(all_edges, neighbors, authors)
    aa_edges = adamic_adar(all_edges, neighbors)
    
    # Distance matrix
    dist_m = fill_matrix(dist, num_authors)
    # Common neighbors matrix
    cn_m = transform_matrix(cn_edges, num_authors)
    # Common neighbors matrix
    aa_m = transform_matrix(aa_edges, num_authors)

    P_train = build_matrix(train_adj_list, author_ind)
    P_test = build_matrix(test_adj_list, author_ind)
    
    max_dist = 4
    neighbor_count = {}
    for d in range(1, max_dist + 1):
        neighbor_count[d] = (gen_neighbor_count(d, dist, num_authors))
    edges_year_dict = {}


    # neighbor_count = np.vstack(neighbor_count)

    # pickle.dump(dist_m, open(category + '_dist_m.pkl', 'wb'))
    # pickle.dump(cn_m, open(category + '_cn_m.pkl', 'wb'))
    # pickle.dump(aa_m, open(category + '_aa_m.pkl', 'wb'))
    # pickle.dump(neighbor_count, open(category + '_neighbor_count.pkl', 'wb'))

    if train:
        edges = all_edges
        actual_m = P_train
        edges_year_dict = build_year_dicts(train_adj_list_with_year)
    else:
        edges = pos_edges
        actual_m = P_test
        edges_year_dict = build_year_dicts(test_adj_list_with_year)
    
    edges = list(edges)
    dist_list, cn_list, aa_list = [], [], []
    n1_list_1, n2_list_1, n3_list_1, n4_list_1 = [], [], [], []
    n1_list_2, n2_list_2, n3_list_2, n4_list_2 = [], [], [], []
    year_list, edge_list, edge_exist_list = [], [], []

    count = 0
    for e in tqdm(edges):
        if aa_m[e[0]][e[1]] > 0.01 and cn_m[e[0]][e[1]] > 0 and \
            neighbor_count[1][e[0]] > 0 and neighbor_count[1][e[1]] > 0 \
            or np.random.rand() > sample:
            dist_list.append(dist_m[e[0]][e[1]])
            edge_list.append(e)
            cn_list.append(cn_m[e[0]][e[1]])
            aa_list.append(aa_m[e[0]][e[1]])
            n1_list_1.append(neighbor_count[1][e[0]])
            n2_list_1.append(neighbor_count[2][e[0]])
            n3_list_1.append(neighbor_count[3][e[0]])
            n4_list_1.append(neighbor_count[4][e[0]])
            n1_list_2.append(neighbor_count[1][e[1]])
            n2_list_2.append(neighbor_count[2][e[1]])
            n3_list_2.append(neighbor_count[3][e[1]])
            n4_list_2.append(neighbor_count[4][e[1]])
            year = None
            if edges_year_dict.has_key(e):
                year = edges_year_dict[e]
            year_list.append(year)
            edge_exist_list.append(actual_m[e[0]][e[1]])
            count += 1
   
    d = {'edge':edge_list, 'common_neighbors':cn_list, 'adamic_adar':aa_list,
        'n1_node1':n1_list_1, 'n2_node1':n2_list_1, 'n3_node1':n3_list_1,
        'n4_node1':n4_list_1, 'n1_node2': n1_list_2, 'n2_node2': n2_list_2,
        'n3_node2': n3_list_2, 'n4_node2': n4_list_2, 'year':year_list, 
        'target': edge_exist_list}    
    df = pd.DataFrame(data=d)
    cols = ['edge', 'common_neighbors', 'adamic_adar', 'n1_node1', 'n2_node1',
           'n3_node1', 'n4_node1', 'n1_node2', 'n2_node2', 'n3_node2', 'n4_node2',
           'year', 'target']
    df = df[cols]
    
    if write:
        if train:
            df.to_csv(category + "_train_df.csv", index=False)
        else:
            df.to_csv(category + "_test_df.csv", index=False)
    
    return df