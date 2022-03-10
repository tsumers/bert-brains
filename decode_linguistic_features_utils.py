import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
import csv

def LoadLingFeatures(args,feature_name):
    #Load linguistic features
    with open(f'data/{args.STIMULUS}/tr_tokens_new.csv','r') as f:
        reader = csv.reader(f)
        ling_file = [row for row in reader]
        ling_head = ling_file[0]
        ling_text = ling_file[1:]
    features = [[element[1:-1] for element in row[ling_head.index(feature_name)][1:-1].split(', ')]\
                if len(row[ling_head.index(feature_name)]) >0 else [] for row in ling_text]
    feature_ids = np.array([feature[0] != '' for feature in features])
    print(f'# of TRs with ling features: {np.sum(feature_ids==1)}')
    print(f'# of TRs without ling features: {np.sum(feature_ids==0)}')
    return features,feature_ids

#def MySampleWeight(y,num_classes=2):
#    neg_samples = y==0
#    pos_samples = y==1
#    sample_weight = np.empty(len(y))
#    sample_weight[pos_samples] = len(y)/(num_classes*pos_samples.sum())
#    sample_weight[neg_samples] = len(y)/(num_classes*neg_samples.sum())
#    return sample_weight

def split_dataset(X,y,split_num,block_cv=False):
    assert X.shape[0] == y.shape[0], 'Shape mismatch between X and y; make sure both are np arrays'
    random_ids = np.random.permutation(X.shape[0])
    if block_cv:
        random_ids = np.arange(X.shape[0])
    batch_size = X.shape[0] // split_num
    train_data = []
    eval_data = []
    for i in range(split_num):
        if i == split_num-1:
            eval_ids = random_ids[batch_size*i:]
        else:
            eval_ids = random_ids[batch_size*i:batch_size*(i+1)]
        train_ids = np.array([element for element in random_ids if element not in eval_ids])
        assert len(eval_ids) +  len(train_ids) == X.shape[0]
        train_X = X[train_ids]
        eval_X = X[eval_ids]
        train_y = y[train_ids]
        eval_y = y[eval_ids]
        train_data.append([train_X,train_y])
        eval_data.append([eval_X,eval_y])
    return train_data,eval_data

def MyCrossValidation(X,y,model,split_num,block_cv,score_balance=False):
    val_score_list = []
    params= []
    train_data,eval_data = split_dataset(X,y,split_num,block_cv)
    while ~np.all([split_data[1].sum()>0 for split_data in train_data]):
        train_data,eval_data = split_dataset(X,y,split_num,block_cv)
    for i in range(split_num):
        train_X = train_data[i][0]
        train_y = train_data[i][1]
        eval_X = eval_data[i][0]
        eval_y = eval_data[i][1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.fit(train_X,train_y)
        if score_balance:
            sample_weight = compute_sample_weight('balanced',eval_y)
            #assert np.all(sample_weight==MySampleWeight(eval_y))
            score = model.score(eval_X,eval_y,sample_weight=sample_weight)
            assert score==np.average(model.predict(eval_X)==eval_y,weights=sample_weight)
        else:
            score = model.score(eval_X,eval_y)
            assert score==np.average(model.predict(eval_X)==eval_y)
        val_score_list.append(score)
        params.append(model.coef_[0])
    return np.array(val_score_list),np.array(params)

def MyOuterCrossValidation(X,y,alphas,reg_type,outer_split_num,inner_split_num,block_cv,train_balance=False,score_balance=False,solver='liblinear'):
    train_data,test_data = split_dataset(X,y,outer_split_num,block_cv)
    while ~np.all([split_data[1].sum()>0 for split_data in train_data]):
        train_data,test_data = split_dataset(X,y,outer_split_num,block_cv)
    score_list = []
    max_alpha_list = []
    coef_list = []
    intercept_list = []
    for i in range(outer_split_num):
        train_X = train_data[i][0]
        train_y = train_data[i][1]
        test_X = test_data[i][0]
        test_y = test_data[i][1]
        val_results = np.empty((len(alphas),inner_split_num))
        for alpha_id, alpha in enumerate(alphas):
            #print(alpha)
            if reg_type=='poisson':
                clf = PoissonRegressor(alpha=alpha)
            elif reg_type=='logistic':
                if train_balance:
                    clf = LogisticRegression(penalty='l2',solver=solver,C=alpha,max_iter=10000,class_weight='balanced')
                else:
                    clf = LogisticRegression(penalty='l2',solver=solver,C=alpha,max_iter=10000)
            scores, _ = MyCrossValidation(train_X,train_y,clf,inner_split_num,block_cv,score_balance)
            val_results[alpha_id,:] = scores

        val_results_ave = val_results.mean(axis=-1)
        max_alpha = alphas[val_results_ave.argmax()]
        max_alpha_list.append(max_alpha)
        #if val_results_ave.argmax()==0 or val_results_ave.argmax()==len(alphas)-1:
        #    assert val_results_ave[val_results_ave.argmax()]==val_results_ave[val_results_ave.argsort()[-2]]

        if reg_type=='poisson':
            new_clf = PoissonRegressor(alpha=max_alpha)
        elif reg_type=='logistic':
            if train_balance:
                new_clf = LogisticRegression(penalty='l2',solver=solver,C=max_alpha,max_iter=10000,class_weight='balanced')
            else:
                new_clf = LogisticRegression(penalty='l2',solver=solver,C=max_alpha,max_iter=10000)

        new_clf.fit(train_X,train_y)
        coef_list.append(new_clf.coef_[0])
        intercept_list.append(new_clf.intercept_[0])
        if score_balance:
            sample_weight = compute_sample_weight('balanced',test_y)
            #assert np.all(sample_weight==MySampleWeight(eval_y))
            score = new_clf.score(test_X,test_y,sample_weight=sample_weight)
            assert score==np.average(new_clf.predict(test_X)==test_y,weights=sample_weight)
        else:
            score = new_clf.score(test_X,test_y)
            assert score==np.average(new_clf.predict(test_X)==test_y)
        score_list.append(score)
    return np.array(score_list),np.array(max_alpha_list),np.array(coef_list),np.array(intercept_list)
