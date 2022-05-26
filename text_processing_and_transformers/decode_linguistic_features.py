import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import collections
import os
import argparse
import warnings
from decode_linguistic_features_utils import MyOuterCrossValidation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--STIMULUS', type=str, required=True)
    parser.add_argument('--MODEL', type=str, default='bert-base-uncased')
    parser.add_argument('--outer_split_num', type=int, default=5)
    parser.add_argument('--inner_split_num', type=int, default=5)
    parser.add_argument('--reg_type', type=str, choices=['poisson','logistic'], default='logistic')
    parser.add_argument('--train_balance',dest='train_balance',action='store_true',default=False)
    parser.add_argument('--score_balance',dest='score_balance',action='store_true',default=False)
    parser.add_argument('--block_cv',dest='block_cv',action='store_true',default=False)
    parser.add_argument('--solver', type=str, choices=['liblinear','lbfgs','newton-cg','sag','saga'], default='liblinear')
    args = parser.parse_args()
    print(f'Running with args: {args}')

    if args.block_cv:
        block_cv_id = '_block_cv'
    else:
        block_cv_id = ''

    if args.train_balance:
        train_balance_id = '_train_balanced'
    else:
        train_balance_id = ''

    if args.score_balance:
        score_balance_id = '_score_balanced'
    else:
        score_balance_id = ''

    loaded_data = np.load(f'data/{args.STIMULUS}/{args.STIMULUS}_ling_features.npz')
    ling_data = loaded_data['ling_features']
    ling_labels = list(loaded_data['dep_list'])+list(loaded_data['pos_list'])
    #print(ling_labels)

    z_reps_list = []
    for layer_num in range(12):
        z_rep_fname = f'data/{args.STIMULUS}/{args.MODEL}/raw_embeddings/{args.STIMULUS}_{args.MODEL}_layer_{layer_num}_z_representations.npy'
        loaded_data = np.load(z_rep_fname)
        z_reps_list.append(loaded_data)

    alphas = np.logspace(-30,30,num=11)
    results_all = np.empty((12,12,len(ling_labels),args.outer_split_num))
    max_alphas_all = np.empty((12,12,len(ling_labels),args.outer_split_num))
    coef_all = np.empty((12,12,len(ling_labels),args.outer_split_num,64))
    intercept_all = np.empty((12,12,len(ling_labels),args.outer_split_num))
    for layer_id in range(12):
        print(f'Processing layer {layer_id}')
        for head_id in range(12):
            #print(f'Processing head {head_id}')
            x = z_reps_list[layer_id][:,64*head_id:64*(head_id+1)]
            if args.reg_type=='poisson':
                features = ling_data
            elif args.reg_type=='logistic':
                features = ling_data>0
            assert len(ling_labels)==features.shape[1]
            for label_id,label in enumerate(ling_labels):
                #print(label)
                y = features[:,label_id]
                score,max_alpha,coef,intercept = MyOuterCrossValidation(x,y,alphas,args.reg_type,
                                                                        args.outer_split_num,args.inner_split_num,
                                                                        args.block_cv,args.train_balance,args.score_balance,
                                                                        args.solver)
                results_all[layer_id,head_id,label_id,:] = score
                max_alphas_all[layer_id,head_id,label_id,:] = max_alpha
                coef_all[layer_id,head_id,label_id,:,:] = coef
                intercept_all[layer_id,head_id,label_id,:] = intercept

    outfile = f'decoding_results_indiv_{args.MODEL}_{args.STIMULUS}_{args.reg_type}_'\
                +f'{args.outer_split_num}_{args.inner_split_num}'\
                +f'{block_cv_id}{train_balance_id}{score_balance_id}_nested_{args.solver}'
    np.savez(outfile,labels=ling_labels,acc=results_all.mean(axis=-1),
            acc_all=results_all,max_alphas_all=max_alphas_all,
            weight=coef_all.mean(axis=-2),weight_all=coef_all,bias_all=intercept_all)
