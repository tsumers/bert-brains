import numpy as np 
import nibabel as nib 
import matplotlib
from seaborn.miscplot import palplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import nilearn.plotting as niplt
import os 
import seaborn as sns 
from scipy.special import softmax 
import pandas as pd 
import sys 

#layer_or_z=int(sys.argv[1])


#excluded 296
black_subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-127', 'sub-279', 
'sub-315', 'sub-314']
slumlordreach_subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']


def get_result_dataset(dataset,rep_name,normalize_isc=True):
    results=[]
    for story in dataset:
        results_story=[]
        if story=='black':
            subs=black_subs
        if story=='slumlordreach':
            subs=slumlordreach_subs
        result_dir='/jukebox/griffiths/bert-brains/results/'+story+"/encoding-"+rep_name+"/"
        roi_mapping=np.load('/jukebox/griffiths/bert-brains/'+story+'_data/Parcel2ROI_new.npy')
        for sub in subs:
            results_sub_parcels=np.arctanh(np.load(result_dir+sub+"_parcelwise_results_banded_ridge.npy")[:,3])
            noise_ceiling_parcels=np.arctanh(np.load('/jukebox/griffiths/bert-brains/'+story+"_data/isc/"+sub+'.npy'))
            results_sub_parcels[np.isnan(results_sub_parcels)]=0.0
            noise_ceiling_parcels[np.isnan(noise_ceiling_parcels)]=0.0
            results_sub=np.zeros((22,)) 
            noise_sub=np.zeros((22,))
            
            for roi in range(22):
                results_sub[roi]=results_sub_parcels[np.where(roi_mapping==roi+1)].mean() 
                noise_sub[roi]=noise_ceiling_parcels[np.where(roi_mapping==roi+1)].mean()
            #results_story.append(-1*np.log(results_sub/noise_sub))
            if normalize_isc:
                r=(results_sub/noise_sub)*100.0 
                r[r>100.0]=100.0
                results_story.append((results_sub/noise_sub)*100.0)
            else:
                results_story.append((results_sub))
        results_story=np.asarray(results_story)
        results.append(results_story)
    results=np.asarray(results)
    results=np.concatenate(results,axis=0)
    results[np.isnan(results)]=0.0
    return results 
    #return results*-1 
    #if normalize_isc:
    #    return (np.concatenate(results,axis=0)/noise_ceiling)*100.0
    #else:
    #    return np.concatenate(results,axis=0)

rois=['PostTemp','AntTemp','AngG','IFG','MFG','IFGorb','vmPFC','dmPFC','PMC','HG','V1']
#rois=['HG','PostTemp','AntTemp','AngG','IFG','IFGorb','MFG','vmPFC','dmPFC','PMC']

roi_names=['L_'+roi for roi in rois]+['R_'+roi for roi in rois] 

raw_data_layers=np.asarray([get_result_dataset(['black','slumlordreach'],'layer_'+str(i)+"_activations") for i in range(12)])
raw_data_z=np.asarray([get_result_dataset(['black','slumlordreach'],'layer_'+str(i)+"_z_representations") for i in range(12)])

roi=[] 
performance=[]
roi_color=[]
layers=[] 
rep_type=[]
for layer_number in range(12):
    curr_layer=raw_data_layers[layer_number]
    curr_z=raw_data_z[layer_number]

    for i in range(curr_layer.shape[0]):
        for j in range(11,22):

            #hemi.append('L' if j<10 else 'R')
            r_layer=curr_layer[i,j]
            r_z=curr_z[i,j]
            
            roi.append(rois[j%11])
            roi_color.append(rois[j%11])
            performance.append(r_layer)
            layers.append(layer_number)
            rep_type.append('Embedding')

            roi.append(rois[j%11])
            roi_color.append(rois[j%11])
            performance.append(r_z)
            layers.append(layer_number)
            rep_type.append('Transformation')



roi=np.asarray(roi)
performance=np.asarray(performance)
roi_color=np.asarray(roi_color)
layers=np.asarray(layers)
rep_type=np.asarray(rep_type)

#hemi=np.asarray(hemi)
df=pd.DataFrame(dict(performance=performance,roi=roi,roi_color=roi_color,layer=layers,rep_type=rep_type))


plt.figure()
colors=list(sns.mpl_palette('tab10',n_colors=10).as_hex())+['#000000'] 
pal=sns.color_palette(colors) 

#pal2=sns.mpl_palette('cividis',n_colors=12) 
#pal2=sns.mpl_palette('bwr',n_colors=2) 
pal2=sns.color_palette(['blue','red'])

roi_order=['HG','PostTemp','AntTemp','AngG','IFG',
'IFGorb','MFG','vmPFC','dmPFC','PMC']


order=roi_order 

#v=sns.violinplot(data=df,x='roi',y='performance',zorder=0,inner=None,linewidth=1,cut=0,hue='roi',alpha=0.2)
#sns.stripplot(data=df,x='roi',y='performance',dodge=0.4,zorder=1,size=2,order=order,hue='layer',palette=pal2)
g=sns.FacetGrid(data=df,col='roi',hue='rep_type',palette=pal2,col_order=order,col_wrap=5,sharex=True,sharey=True)
g.map(sns.pointplot,'layer','performance',dodge=8.0,join=True,ci=95,n_boot=10000,capsize=0,zorder=2,estimator=np.median,err_kws = {"alpha": .5},errwidth=0.8,s=3)
#g=sns.catplot(data=df,x='roi',y='performance',hue='rep_type',col='roi',col_order=order,col_wrap=5,sharex=True,sharey=True,palette=pal2,kind='point',n_boot=10000,capsize=0,errwidth=0.8,s=3)
#sns.boxplot(x='roi',y='performance',data=df,saturation=0.35,fliersize=0) 
#plt.setp(g.collections, alpha=.5) #for the markers
#plt.setp(g.lines, alpha=.3)       #for the lines

#plt.legend(['L','R'],['C0','C1'])
plt.axhline(y=0,color='black')
#plt.legend([])
sns.despine(top=True,right=True,left=False,bottom=False)
colors2=list(sns.mpl_palette('cividis',n_colors=12).as_hex()) 

index=0
for index in range(10):
    g.axes[index].set_title(order[index],color=colors[index])
    g.axes[index].set_xticklabels(['Layer '+str(i) for i in range(1,13)],rotation=90)


my_colors=[]
"""
tick_labels=[] 
for i in range(len(rois))[idx_1:(idx_1+1)]:
    my_colors+=colors2 
    tick_labels+=["Layer "+str(j) for j in range(1,13)] 

plt.xticks(ticks=list(range(len(tick_labels))),labels=tick_labels,rotation=90)
"""
#for sep in range(1,11):
#    plt.axvline(x=sep*11,linestyle='--',color='black')

idx=0
#for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
#    ticklabel.set_color(tickcolor)
#    idx+=1
#plt.xlabel("")  
#plt.ylabel("")
plt.ylim(0,30) 
#plt.gca().get_legend().remove()
plt.savefig('plots/layer_embeddings_plots/full_facet_plot_right.svg',format='svg')  