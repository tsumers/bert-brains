import numpy as np
import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
import collections
import itertools
import os
import argparse
import spacy
from decode_linguistic_features_utils import LoadLingFeatures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--STIMULUS', type=str, required=True, choices=['black','slumlordreach'])
    args = parser.parse_args()
    print(f'Running with args: {args}')

    nlp = spacy.load('en_core_web_lg')

    # Extract POS, TAG, DEP, etc. and save the data to ling_features.csv
    with open(f'data/{args.STIMULUS}/align.csv','r') as f:
        reader = csv.reader(f)
        token_list = [row[0] for row in reader]
    doc = nlp(' '.join(token_list))
    sent_id = 0
    ling_features = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != 'SPACE':
                ling_features.append([sent_id,token.text,token.lemma_,token.pos_,token.tag_,token.dep_,token.head.text,abs(token.i-token.head.i)])
        sent_id += 1

    ling_features = pd.DataFrame(ling_features,columns=["index","token","lemma","pos","tag","dep_rel","dep_head","dep_distance"])
    ling_features.to_csv(f'data/{args.STIMULUS}/ling_features.csv')

    # Align TRs
    # Load ling_features.csv
    with open(f'data/{args.STIMULUS}/ling_features.csv', 'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
        head = file[0]
        text = file[1:]

    # Load tr_tokens.csv
    with open(f'data/{args.STIMULUS}/tr_tokens.csv', 'r') as f:
        reader = csv.reader(f)
        TR_file = [row for row in reader]
        TR_head = TR_file[0]
        TR_text = TR_file[1:]

    # Align the tr data
    current_id = 0
    new_data = []
    for TR_line in TR_text:
        TR_tokens = TR_line[TR_head.index('tokens')]
        lemma_list = []
        pos_list = []
        tag_list = []
        dep_rel_list = []
        dep_head_list = []
        dep_dist_list = []
        if TR_tokens != "":
            token_list = [token.text for token in nlp(TR_tokens)]
            for token in token_list:
                line = text[current_id]
                lemma_list.append(line[head.index('lemma')])
                pos_list.append(line[head.index('pos')])
                tag_list.append(line[head.index('tag')])
                dep_rel_list.append(line[head.index('dep_rel')])
                dep_head_list.append(line[head.index('dep_head')])
                dep_dist_list.append(line[head.index('dep_distance')])
                current_id += 1
        TR_line.extend([lemma_list,pos_list,tag_list,dep_rel_list,dep_head_list,dep_dist_list])
        new_data.append(TR_line)
    TR_head.extend(["lemma","pos","tag","dep_rel","dep_head","dep_distance"])
    new_data = pd.DataFrame(new_data,columns=TR_head)

    new_data.to_csv(f'data/{args.STIMULUS}/tr_tokens_new.csv')
    new_data.to_pickle(f'data/{args.STIMULUS}/tr_tokens_new.pkl')

    in_file = f'data/{args.STIMULUS}/tr_tokens_new.csv'
    print(f'Loading {in_file}')
    with open(in_file,'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
        head = file[0]
        text = file[1:]
    with open(f'data/{args.STIMULUS}/dep_dist.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['tr','label','dep_dist'])
        for tr_id,line in enumerate(text):
            assert len(line[head.index('dep_rel')][1:-1].split(','))==len(line[head.index('dep_distance')][1:-1].split(','))
            for dep,dep_dist in zip(line[head.index('dep_rel')][1:-1].split(','),line[head.index('dep_distance')][1:-1].split(',')):
                writer.writerow([tr_id,dep.strip(" '"),dep_dist.strip(" '")])
            #new_data.extend([[tr_id,pos,dep_dist] for pos,dep_dist in
            #                 zip(line[head.index('pos')][1:-1].split(','),line[head.index('dep_distance')][1:-1].split(','))])
            #new_data.extend([[tr_id,dep,dep_dist] for dep,dep_dist in
            #                 zip(line[head.index('dep_rel')][1:-1].split(','),line[head.index('dep_distance')][1:-1].split(','))])
    #dep_dist_df = pd.DataFrame(new_data,columns=['tr','label','dep_dist'])
    #dep_dist_df.to_csv(f'data/{args.STIMULUS}/dep_dist.csv',index=False)

    dep_features, dep_feature_ids = LoadLingFeatures(args,'dep_rel')
    pos_features, pos_feature_ids = LoadLingFeatures(args,'pos')

    od_dep = collections.OrderedDict(collections.Counter(list(itertools.chain.from_iterable(dep_features))))
    od_dep = collections.OrderedDict(sorted(od_dep.items(),key=lambda x: x[1], reverse=True))
    print(od_dep)

    od_pos = collections.OrderedDict(collections.Counter(list(itertools.chain.from_iterable(pos_features))))
    od_pos = collections.OrderedDict(sorted(od_pos.items(),key=lambda x: x[1], reverse=True))
    print(od_pos)

    dep_list = ['nsubj', 'ROOT', 'advmod', 'prep', 'det',
                'pobj', 'aux', 'dobj', 'cc', 'ccomp',
                'amod', 'compound', 'acomp', 'poss', 'xcomp',
                'conj', 'relcl', 'attr', 'mark', 'npadvmod',
                'advcl', 'neg', 'prt', 'nummod', 'intj']
    pos_list = ['PRON', 'VERB', 'NOUN', 'DET', 'AUX', 'ADP', 'ADV',
                'CCONJ', 'ADJ', 'PART', 'PROPN', 'SCONJ', 'NUM', 'INTJ']

    ling_features = np.array([[np.sum([element==dep_label for element in dep_tr_data]) for dep_label in dep_list]
                                +[np.sum([element==pos_label for element in pos_tr_data]) for pos_label in pos_list]
                                for dep_tr_data,pos_tr_data in zip(dep_features,pos_features)])

    np.savez(f'{args.STIMULUS}_ling_features.npz',ling_features=ling_features,dep_list=dep_list,pos_list=pos_list)
