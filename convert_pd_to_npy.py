import pandas as pd 
import numpy as np 
import pickle
to_convert=['data/21st_year/gpt2/syntactic_analyses/21st_year_gpt2_syntactic_complexity_L-inf.pkl',
'data/21st_year/gpt2/syntactic_analyses/21st_year_gpt2_syntactic_distance.pkl','data/21st_year/gpt2/syntactic_analyses/21st_year_gpt2_semantic_composition_max_l2.pkl']

for fname in to_convert:
	df=pickle.load(open(fname,'rb'))
	print(df)
	a=df.to_numpy()[:,0]
	buf=[]
	size=len(a[int(len(a)/2)])
	for row in a:
		if row==None:
			buf.append(np.ones(size,)*np.nan)
		else:
			buf.append(np.asarray(row))
	buf=np.asarray(buf)
	print(buf.shape)
	np.save(fname[:-3]+'npy',buf)

