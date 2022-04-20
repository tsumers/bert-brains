
import numpy as np 
from scipy.stats import pearsonr 
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression, Ridge 
import sys 
from sklearn.decomposition import PCA 

d='/jukebox/griffiths/bert-brains/'
dataset=sys.argv[1]
sub=sys.argv[2]





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

sub_idx=subs.index(sub)
data_dir=d+dataset+'_data/'  
full_data=[np.load(data_dir+sub+"_parcelwise_data.npy") for sub in subs]
heldout_average=np.mean([full_data[i] for i in range(len(full_data))],axis=0)
skf=KFold(n_splits=3,shuffle=False) 
#regressor=np.load(data_dir+"silent_vector.npy")
nuisance=np.load(data_dir+"nuisance_features.npy")
pca=PCA(n_components=1)
X0=pca.fit_transform(nuisance).reshape((-1,1))
X0=(X0-X0.mean(axis=0))/(X0.std(axis=0,ddof=1)+1e-9) 
#X0=np.load(data_dir+"nuisance_features.npy")





print(sub)
#isc_data=np.zeros((1000,))
#heldout_average=np.mean([full_data[i] for i in range(len(full_data)) if i!=sub_idx],axis=0)
#heldout_average=np.mean([full_data[i] for i in range(len(full_data))],axis=0)
#sub_data=full_data[sub_idx]
#isc_data=np.asarray([pearsonr(sub_data[p,:],heldout_average[p,:])[0] for p in range(1000)])
#isc_data[np.isnan(isc_data)]=0.0
sub_data=full_data[sub_idx]
isc_data=np.zeros((1000,3))

for p in range(1000):
    x=sub_data[p,:].reshape((-1,1))
    y=heldout_average[p,:]
    #model=LinearRegression()
    #model.fit(regressor,x) 
    #x=x-model.predict(regressor)
    
    
    #resid_model=LinearRegression(normalize=False)
    #resid_model.fit(X0,x) 
    #x=x-resid_model.predict(X0).reshape((-1,1))  

    

    test_idx=0
    for train_index,test_index in skf.split(x):
        isc_data[p,test_idx]=pearsonr(x[test_index,0],y[test_index])[0]
        test_idx+=1 
isc_data[np.isnan(isc_data)]=0.0 
isc_data=np.mean(isc_data,axis=1)
print(isc_data.shape)

    


np.save(data_dir+"isc/"+sub+".npy",isc_data) 
