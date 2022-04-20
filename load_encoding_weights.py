import numpy as np 
import os 
import pickle
def load_weights(dataset,collapse_delay=True):
	d="/jukebox/griffiths/bert-brains/"
	if dataset=='slumlordreach':
		subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
		'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
	elif dataset=='black':
		subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 'sub-291', 
		'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
		'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
		'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
		'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
		'sub-315', 'sub-314']
	else:
		return None 
	"""
	layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)] 
	layer_names+=['layer_'+str(i)+"_z_representations" for i in range(0,12)]
	layer_prefix=d+'code/bert-brains/data/'+dataset+'/bert-base-uncased/raw_embeddings/'
	save_prefix=d+"results/"+dataset+"/"
	layer_dirs=[layer_prefix+dataset+"_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
	save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]
	"""
	save_prefix=d+"results/"+dataset+"/"
	layer_names=[]
	layer_dirs=[]
	save_dirs=[]
	layer_prefix=d+'code/bert-brains/data/'+dataset+'/bert-base-uncased/syntactic_analyses/'
	for fname in os.listdir(layer_prefix):
		if 'bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat' in fname:
			layer_names.append(fname[:-4]) 
			layer_dirs.append(layer_prefix+fname)
			save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/")

	layer_names+=['ling_features']
	layer_dirs+=[d+"code/bert-brains/data/"+dataset+"/ling_features.npy"]
	save_dirs+=[d+"results/"+dataset+"/encoding-ling_features/"]

	layer_names+=['glove']
	layer_dirs+=[d+'code/bert-brains/data/'+dataset+'/bert-base-uncased/raw_embeddings/'+dataset+'_bert-base-uncased_layer_0_glove.npy']
	save_dirs+=[d+"results/"+dataset+"/encoding-glove/"]

	layer_names+=['full_z']
	layer_dirs+=['/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/full_z_representations.npy']
	save_dirs+=[d+"results/"+dataset+"/encoding_full_z/"]

	layer_names+=['full']
	layer_dirs+=['/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/full_layer_embeddings.npy']
	save_dirs+=[d+"results/"+dataset+"/encoding_full/"]



	
	assert len(layer_names)==len(layer_dirs)
	assert len(save_dirs)==len(layer_names)

	return_dict={}
	for i in range(len(layer_names)):
		print(i,len(layer_names),layer_names[i])
		layer_name=layer_names[i]
		save_dir=save_dirs[i]
		layer_dir=layer_dirs[i]

		load_features=np.load(layer_dir,allow_pickle=True)
		raw_features=[]
		for i in range(load_features.shape[0]):
			if load_features[i] is not None and len(load_features[i])>0:
				if 'semantic_composition' in layer_dir:
					raw_features.append(load_features[i][0])
				else:
					raw_features.append(load_features[i])
		raw_features=np.vstack(raw_features)
		num_units=raw_features.shape[1]

		total_weights=[]
		if layer_name=='full':
			sub_list=subs[:11]
		else:
			sub_list=subs

		for sub in sub_list:
			weights_sub=np.load(save_dir+sub+"_encoding_weights.npy")[:,0,:]
			if 'full' in layer_name and len(weights_sub.shape)==3:
				weights_sub=weights_sub[:,0,:]
			if collapse_delay:
				assert weights_sub.shape[1]%4==0
				by_weights=[]
				for d in range(4):
					by_weights.append(weights_sub[:,int(d*num_units):int((d+1)*num_units)])
				by_weights=np.asarray(by_weights)
				total_weights.append(np.mean(by_weights,axis=0))
			else:
				total_weights.append(weights_sub)
		total_weights=np.asarray(total_weights)
		return_dict[layer_name]=total_weights
	return return_dict

for dataset in ['black']:
	x=load_weights(dataset)
	fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/'+dataset+'/bert-base-uncased/encoding_weights.npy'
	np.savez(fname,**x)






