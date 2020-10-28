import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
d='/jukebox/griffiths/bert-brains/'
from brainiak.funcalign.srm import SRM
from brainiak.fcma.util import compute_correlation
from scipy.stats import stats
import sys 
import os

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']

# time segment matching experiment. 
def time_segment_matching_accuracy(data, tst_subj,win_size=9): 
    nsubjs = len(data)
    (ndim, nsample) = data[0].shape
    nseg = nsample - win_size 
    # mysseg prediction prediction
    trn_data = np.zeros((ndim*win_size, nseg),order='f')
    # the trn data also include the tst data, but will be subtracted when 
    # calculating A
    for m in range(nsubjs): 
        for w in range(win_size):
            trn_data[w*ndim:(w+1)*ndim,:] += data[m][:,w:(w+nseg)]
    tst_data = np.zeros((ndim*win_size, nseg),order='f')
    for w in range(win_size):
        tst_data[w*ndim:(w+1)*ndim,:] = data[tst_subj][:,w:(w+nseg)]
    A =  np.nan_to_num(stats.zscore((trn_data - tst_data),axis=0, ddof=1))
    B =  np.nan_to_num(stats.zscore(tst_data,axis=0, ddof=1))
    # compute correlation matrix
    corr_mtx = compute_correlation(B.T,A.T)
    for i in range(nseg):
        for j in range(nseg):
            if abs(i-j)<win_size and i != j :
                corr_mtx[i,j] = -np.inf
    max_idx =  np.argmax(corr_mtx, axis=1)
    return sum(max_idx == range(nseg)) / float(nseg)
def segmentation(num_feats,sub,shuffle=False):
    shared_name = d+"21styear_data/srm_shared_"+str(num_feats)+".npy"
    shared_data=np.load(shared_name) 
    for s in range(shared_data.shape[0]):
        x=np.nan_to_num(stats.zscore(shared_data[s],axis=1,ddof=1))
        if shuffle:
            np.random.shuffle(x)
        shared_data[s] = np.nan_to_num(stats.zscore(shared_data[s],axis=1,ddof=1))
    # run experiment
    accu = time_segment_matching_accuracy(shared_data,sub)
    # return: can also return several values. In that case, the final output will be 
    # a 3D array of tuples
    return accu
"""
mean_acc=[]
f=[]
for num_features in list(range(10,600,10)):
    fname=d+"21styear_data/srm_shared_"+str(num_features)+".npy"
    ts=[]
    if os.path.isfile(fname):
        for sub in range(len(subs)):
            ts.append(segmentation(num_features,sub))
        mean_acc.append(np.mean(ts))
        f.append(num_features) 

mean_acc=np.asarray(mean_acc)
f=np.asarray(f)
np.save('time_segment_accuracies.npy',mean_acc)
np.save('time_segment_features.npy',f)
"""

num_features=10
fname=d+"21styear_data/srm_shared_"+str(num_features)+".npy"
mean_acc=[]
for i in range(1000):
    print(i)
    ts=[]
    for sub in range(len(subs)):
        ts.append(segmentation(num_features,sub,shuffle=True))
    mean_acc.append(np.mean(ts))
print(np.mean(mean_acc))
#sub=int(sys.argv[2])
#print(segmentation(num_features,sub))
