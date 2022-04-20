import os 

for dataset in ['black','slumlordreach']:
    
    for i in range(12):
        d1='/jukebox/griffiths/bert-brains/results/'+dataset+"/encoding-_layer_"+str(i)+"_z_representations/"
        d2='/jukebox/griffiths/bert-brains/results/'+dataset+"/encoding-layer_"+str(i)+"_z_representations/"
        os.system('cp -v '+d1+"* "+d2)