import numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors

d='/jukebox/griffiths/bert-brains/'

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']
data_dir=d+'21styear_data/'
if not(os.path.isdir(data_dir+"functional_spaces/")):
	os.mkdir(data_dir+"functional_spaces/")
weights=np.load(data_dir+"srm_weights.npy")

for i,sub in enumerate(subs):
	print(i)
	X=weights[i]
	dim=X.shape[1]
	good_voxels=(np.sum(X==0,axis=1)!=dim)
	good_indices=np.arange(X.shape[0])[good_voxels]
	bad_voxels=(np.sum(X==0,axis=1)==dim)
	bad_indices=np.arange(X.shape[0])[bad_voxels]
	short_X=X[good_voxels]
	nn=NearestNeighbors(n_neighbors=342,leaf_size=2000,metric='cosine')
	nn.fit(short_X)
	short_space=nn.kneighbors(return_distance=False)
	space=np.zeros((X.shape[0],343))
	space[:,0]=np.arange(X.shape[0])
	space[good_voxels,1:]=good_indices[short_space]
	space[bad_voxels,:]=-1.0
	output_name=data_dir+"functional_spaces/"+sub+".npy"
	np.save(output_name,space)


