import numpy as np 
import sys 
import os
prefix='/jukebox/griffiths/bert-brains/'
name=sys.argv[1]
master_participant_list=np.genfromtxt('/jukebox/hasson/snastase/narratives/participants.tsv',delimiter='\t',dtype=str)
if not(os.path.isdir(prefix+name+"_data/")):
	os.mkdir(prefix+name+"_data/")

for i in range(1,master_participant_list.shape[0]):
	if name in master_participant_list[i,3]:
		subj_id=master_participant_list[i,0]
		fname='/jukebox/hasson/snastase/narratives/derivatives/afni-nosmooth/'+subj_id+'/func/'+subj_id+'_task-'+name+'_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz'
		fname2=prefix+name+"_data/"+subj_id+".nii.gz"
		os.system('cp -v '+fname+" "+fname2)

