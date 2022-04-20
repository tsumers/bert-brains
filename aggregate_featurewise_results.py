import os 
import numpy as np 
d='/jukebox/griffiths/bert-brains/'
jobs_per_sub=50 
numbers=list(range(0,1001,jobs_per_sub))

for dataset in ['slumlordreach','black']:

	#Slumlordreach 
	if dataset=='slumlordreach': 

		subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
		'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
		data_dir=d+"slumlordreach_data/"
		save_dir=d+"results/slumlordreach/encoding_featurewise/"



	#Black 
	else:
		subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
		'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
		'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
		'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
			'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
			'sub-315', 'sub-314'] 

		data_dir=d+"black_data/"
		save_dir=d+"results/black/encoding_featurewise/"
	write_dir=d+"results/"+dataset+"/encoding_featurewise/" 
	for sub in subs:
		num_idx=0
		buffer=[]
		while num_idx<len(numbers)-1:
			lower_p=numbers[num_idx]
			upper_p=numbers[num_idx+1]
			buffer.append(np.load(save_dir+sub+"_partial_headwise_result_"+str(lower_p)+"_"+str(upper_p)+".npy"))
			num_idx+=1
		aggregate=np.concatenate(buffer,axis=0) 
		
		print(write_dir+sub+"_parcelwise_results.npy",aggregate.shape)
		np.save(write_dir+sub+"_parcelwise_results.npy",aggregate) 
				


	

