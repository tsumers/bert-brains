# bert-brains

### Preprocessing
`narratives-transcript-processing.ipynb`: Used to pre-process transcripts from the `Narratives` dataset. Outputs (1) phonemes and associated nuisance variables for regression analyses; (2) TR-aligned tokens for use with Transformer notebooks.

### Transformers
`transformer-representations.ipynb`: the primary notebook used to generate Transformer representations (embeddings, transformations) for regression analyses. 
`transformer-utils.py`: functionality for extracting various Transformer representations, including some experimental metrics that were used in the paper. 
`transformer-transformation-magnitudes.ipynb`: small script used to produce transformation magnitudes from the transformations themselves.

### data_moving 
acquire_weight_mats.py: Save all ridge regression weights across subjects into one npy file. 
aggregate_full_layer_results.py: Save full embedding model results into npy files (each job did 50 parcels, so we aggregated outputs across jobs)
create_fmri_dataset.py: Copy data from narratives dataset into lab project folder. 


### Analysis Code
banded_ridge_regression_joint.py: Run encoding model with embedding and transformation jointly fit into different bands (per layer). 
banded_ridge_regression.py: Run encoding model analyses (banded ridge regression) with a given representation as main features 
calculate_boostrap_pvalue.py: Calculate bootstrap pvalue (with FDR corrections) for individual parcels to determine significance. 
compute_isc.py: Compute noise ceilings with intersubject correlation. 
decode_linguistic_features_peters.py: 
decode_linguistic_features.ipynb: 
headwise_banded_ridge_regression.py: Headwise version of the encoding analyses (banded ridge regression, but knock out weights for all heads but one when evaluating a head)



### Visualizations
Head_Dep_Brain_Plots.ipynb: Plots whole brain results for headwise encoding (and dependency encoding).
Layerwise_Plot.ipynb: Plots performance across layers. 


### Slurm Jobs
avg_job.sh: Slurm job corresponding to save_mean_volumes.py. 
boot_job.sh: Slurm job corresponding to calculate_bootstrap_pvalue.py 
dsq-submit.sh: Slurm job that submits a job array specified in a .txt file. 
joblist.txt: Job list for submitting slurm job array


### Misc
fmri_conda_env.yml: yml conda env file for fmri analyses



To delete: 
aggregate_mixing_data.py
aggregate_residual_results.py
aggregate_results.py 
anatomical_searchlight.py
comparison_maps.py
aggregate_featurewise_results.py
encoding_featurewise.py
encoding_headwise.py
encoding_model_fullwidth.py
encoding_model_mixing.py
encoding_model_parcels.py
encoding_model.py
encoding_preference_map.py
encoding_residuals.py
encoding_variance_partitioning.py
