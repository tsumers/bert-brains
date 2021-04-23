import numpy as np 
from scipy.stats import pearsonr 
d='/jukebox/griffiths/bert-brains/'

for dataset in ['slumlordreach','black']:
    print(dataset)
    if dataset=='slumlordreach':
        subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
        'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
    else:
        subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
                'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
                'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
                'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
                    'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
                    'sub-315', 'sub-314']
    
    data_dir=d+dataset+'_data/'  
    full_data=[np.load(data_dir+sub+"_parcelwise_data.npy") for sub in subs]

    for sub_idx,sub in enumerate(subs):
        print(sub)
        isc_data=np.zeros((1000,))
        heldout_average=np.mean([full_data[i] for i in range(len(full_data)) if i!=sub_idx],axis=0)
        sub_data=full_data[sub_idx]
        
        isc_data=np.asarray([pearsonr(sub_data[p,:],heldout_average[p,:])[0] for p in range(1000)])
        isc_data[np.isnan(isc_data)]=0.0

        np.save(data_dir+"isc/"+sub+".npy",isc_data)
