import numpy as np
import os

d='/jukebox/griffiths/bert-brains/'
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)]   
layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/'
save_prefix=d+"results/slumlordreach/"
layer_dirs=[layer_prefix+"slumlordreach_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]
data_dir=d+"slumlordreach_data/" 

layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/syntactic_analyses/'
for fname in os.listdir(layer_prefix):
	layer_names.append(fname[:-4])
	layer_dirs.append(layer_prefix+fname)
	if not(os.path.isdir(save_prefix+'encoding-'+fname[:-4]+"/")):
		os.mkdir(save_prefix+'encoding-'+fname[:-4]+"/")
	save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/")


for layer_dir in layer_dirs:
	load_features=np.load(layer_dir,allow_pickle=True)
	raw_features=[]
	for i in range(load_features.shape[0]):
		if load_features[i] is not None and len(load_features[i])>0:
			if 'semantic_composition_max_l2' in layer_dir:
				raw_features.append(load_features[i][0])
			else:
				raw_features.append(load_features[i])
	raw_features=np.vstack(raw_features)
	print(raw_features)
	print(layer_dir,raw_features.shape)

	
