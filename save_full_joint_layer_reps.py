import numpy as np 

for story in ['black','slumlordreach']:
    fname_all='/jukebox/griffiths/bert-brains/code/bert-brains/data/'+story+'/bert-base-uncased/raw_embeddings/'+story+'_bert-base-uncased_all_layer_activations.npy'
    all_layers=[]
    for layer in range(12):
        embeddings=np.load('data/'+story+'/bert-base-uncased/raw_embeddings/'+story+'_bert-base-uncased_layer_'+str(layer)+'_activations.npy')
        all_layers.append(embeddings)

        transformations=np.load('data/'+story+'/bert-base-uncased/raw_embeddings/'+story+'_bert-base-uncased_layer_'+str(layer)+'_z_representations.npy')
        joint=np.hstack([embeddings,transformations])
        print(joint.shape)
        np.save('data/'+story+'/bert-base-uncased/raw_embeddings/'+story+'_bert-base-uncased_layer_'+str(layer)+'_combined_rep.npy',joint)
    all_layers=np.hstack(all_layers)
    print(all_layers.shape)
    #np.save(fname_all,all_layers)


