import numpy as np
import pickle
import csv
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import collections
import os
import argparse
import math
from scipy.optimize import minimize,fmin_slsqp
from scipy.special import logsumexp

# mixing weights model
def RunMixModel(label):
    print(f'Processing {label}')
    y = CreateLabels(dep_features,label)

    def clf_loss(params):
        print(params)
        alpha = params[0]
        mixing_weight = params[1:]
        model = LogisticRegression(penalty='l2',solver='liblinear',C=alpha)
        
        #Create training/test sets
        np.random.seed(2021)
        random_list = np.random.permutation(vecs.shape[0])
        train_vecs = vecs[random_list[:train_num]]
        train_labels = y[random_list[:train_num]]
        test_vecs = vecs[random_list[train_num:]]
        test_labels = y[random_list[train_num:]]
        
        # Just mixing_weight@vecs should work but just in case...
        train_mixed_features = np.array([mixing_weight@vec for vec in train_vecs])
        model.fit(train_mixed_features,train_labels)
        
        test_mixed_features = np.array([mixing_weight@vec for vec in test_vecs])
        log_prob = model.predict_log_proba(test_mixed_features)
        print(np.sum(params[1:]))
        return np.sum(-test_labels*log_prob[:,1]-(1-test_labels)*log_prob[:,0])
    
    cons=({'type':'eq','fun':lambda x: np.sum(x[1:])-1})
    bounds = [(1e-9,np.inf)]
    bounds.extend([(0.0,np.inf) for _ in range(12)])
    #init_mixing_weight = np.array([1/12 for _ in range(12)])
    init_mixing_weight = np.exp(np.random.randn(12))
    init_mixing_weight/=init_mixing_weight.sum()
    x0 = np.ones(13)
    x0[0] = 1
    x0[1:] = init_mixing_weight
    res=minimize(clf_loss,x0,bounds=bounds,constraints=cons)
    print(res)
    exit()
    return res
    
def LoadLingFeatures(args):
    #Load linguistic features
    with open(f'data/{args.STIMULUS}/tr_tokens_new.csv','r') as f:
        reader = csv.reader(f)
        ling_file = [row for row in reader]
        ling_head = ling_file[0]
        ling_text = ling_file[1:]
    features = [[element[1:-1] for element in row[ling_head.index('dep_rel')][1:-1].split(', ')]\
                if len(row[ling_head.index('dep_rel')]) >0 else [] for row in ling_text]
    feature_ids = np.array([feature[0] != '' for feature in features])
    print(f'# of TRs with ling features: {np.sum(feature_ids==1)}')
    print(f'# of TRs without ling features: {np.sum(feature_ids==0)}')
    return features,feature_ids

def LoadLingFeaturesPOS(args):
    #Load linguistic features
    with open(f'data/{args.STIMULUS}/tr_tokens_new.csv','r') as f:
        reader = csv.reader(f)
        ling_file = [row for row in reader]
        ling_head = ling_file[0]
        ling_text = ling_file[1:]
    features = [[element[1:-1] for element in row[ling_head.index('pos')][1:-1].split(', ')]\
                if len(row[ling_head.index('pos')]) >0 else [] for row in ling_text]
    feature_ids = np.array([feature[0] != '' for feature in features])
    print(f'# of TRs with ling features: {np.sum(feature_ids==1)}')
    print(f'# of TRs without ling features: {np.sum(feature_ids==0)}')
    return features,feature_ids

def CreateLabels(features,label):
    return np.array([1 if label in row else 0 for row in features])

def LoadDataArray(args):
    #Load linguistic features data
    dep_features,dep_feature_ids = LoadLingFeatures(args)
    pos_features,pos_feature_ids = LoadLingFeaturesPOS(args)
    assert np.all(dep_feature_ids==pos_feature_ids)

    #Load representation data
    vecs_list = []
    ids_list = []
    for layer_id in range(12):
        vecs = np.load(f'data/{args.STIMULUS}/{args.MODEL}/syntactic_analyses/'\
                       +f'{args.STIMULUS}_{args.MODEL}_layer_{layer_id}_{args.vec_type}.npy',
                       allow_pickle=True)
        vec_ids = np.array([vec is not None for vec in vecs])

        #Choose TRs that both have z_reps and linguistic features
        assert len(vec_ids) == len(dep_feature_ids)
        ids = np.array([vec_id and feature_id for vec_id, feature_id in
                        zip(vec_ids,dep_feature_ids)])
        new_vecs = np.array([vec for element,vec in zip(ids,vecs) if element])
        vecs_list.append(list(new_vecs))
        ids_list.append(ids)

    # Make sure that we are using the same set of TRs for every layer
    ids_list = np.array(ids_list)
    assert np.all(ids_list.sum(axis=0)%12==0)
    
    print(f'# of TRs used in decoding: {np.sum(ids==1)}')
    new_dep_features = [feature for element,feature in zip(ids,dep_features) if element]
    new_pos_features = [feature for element,feature in zip(ids,pos_features) if element]
    # Transpose so that the shape of the output is (# of TRs, 12, 768)
    out_vecs = np.array(vecs_list).transpose(1,0,2)
    
    assert out_vecs.shape[0] == len(new_dep_features)
    assert out_vecs.shape[0] == len(new_pos_features)
    print(f'Shape of the x array:{out_vecs.shape}')
    return out_vecs,new_dep_features,new_pos_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--STIMULUS', type=str, required=True)
    parser.add_argument('--MODEL', type=str, default='bert-base-uncased')
    parser.add_argument('--split_num', type=int, default=5)
    parser.add_argument('--block_cv', type=bool, default=False)
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--vec_type', type=str, required=True,
                        choices=['z_reps','activations'],
                        help='Type of representations to use: z_reps or activations')

    args = parser.parse_args()
    print(f'running with args: {args}')

    vecs, dep_features, pos_features = LoadDataArray(args)
    dep_list = ['prep','pobj','det','nsubj','amod','dobj',
                'advmod','aux','poss','ccomp','mark','prt']
    pos_list = ['ADV', 'PRON', 'AUX', 'DET', 'NOUN', 'ADP',
                'VERB', 'ADJ', 'PART', 'CCONJ', 'SCONJ', 'PROPN', 'NUM', 'INTJ']
    
    train_num = vecs.shape[0]*9//10
    print(f'# of training examples: {train_num}')
    print(f'# of test examples: {vecs.shape[0]-train_num}')

    RunMixModel(dep_list[0])
    exit()