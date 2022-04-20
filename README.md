# bert-brains

### Preprocessing
`narratives-transcript-processing.ipynb`: Used to pre-process transcripts from the `Narratives` dataset. Outputs (1) phonemes and associated nuisance variables for regression analyses; (2) TR-aligned tokens for use with Transformer notebooks.

### Transformers
`transformer-representations.ipynb`: the primary notebook used to generate Transformer representations (embeddings, transformations) for regression analyses.  
`transformer-transformation-magnitudes.ipynb`: small script used to produce transformation magnitudes from the transformations themselves.  
`transformer-utils.py`: functionality for extracting various Transformer representations, including some experimental metrics that were used in the paper.  

### data_moving/handling
create_fmri_dataset.py: Copy data from narratives dataset into lab project folder. 
ROIs.ipynb: Handling ROI labeling and visualizing. 

### Analysis Code
banded_ridge_regression.py: Run encoding model analyses (banded ridge regression) with a given representation as main features 
calculate_boostrap_pvalue.py: Calculate bootstrap pvalue (with FDR corrections) for individual parcels to determine significance. 
compute_isc.py: Compute noise ceilings with intersubject correlation. 
decode_linguistic_features_peters.py: [FOR TAKA]
decode_linguistic_features.ipynb: [FOR TAKA]
headwise_banded_ridge_regression.py: Headwise version of the encoding analyses (banded ridge regression, but knock out weights for all heads but one when evaluating a head)
run_encoding_models_banded.py: Creates a joblist.txt for banded ridge regression jobs to input as a slurm job array. 
run_encoding_models_headwise_banded.py: Same as above, but for the headwise encoding analyses. 
run_isc_jobs.py: Run multiple subjects' ISC analyses in parallel (slurm jobs)
run_jupyter.sh: Run a jupyter notebook on the cluster. 
run_pvalue_jobs.py: Run bootstrap p-value analyses in parallel. 
run_save_jobs.py: Run jobs in parallel that will save collapsed performance across subjects. 
save_mean_volume.py: Collapse performance across subjects for a specific representation. 



### Visualizations
Head_Dep_Brain_Plots.ipynb: Plots whole brain results for headwise encoding (and dependency encoding).
Layerwise_Plot.ipynb: Plots performance across layers. 
Results_Figures.ipynb: Notebook to plot most main results in the paper. 
View_Encoding_Results.ipynb: Plots whole-brain glass brain results 

### Slurm Jobs
avg_job.sh: Slurm job corresponding to save_mean_volumes.py.
boot_job.sh: Slurm job corresponding to calculate_bootstrap_pvalue.py
dsq-submit.sh: Slurm job that submits a job array specified in a .txt file.
joblist.txt: Job list for submitting slurm job array


### Misc
fmri_conda_env.yml: yml conda env file for fmri analyses



To delete:
aggregate_mixing_data.py
aggregate_full_layer_results.py
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
load_encoding_weights.py
move_z_results.py
multilayer_results.ipynb
nn_job.sh 
run_comparison_jobs.py
run_encoding_models_featurewise.py
run_encoding_models_headwise.py
run_encoding_models_mixing.py
run_encoding_models_parcels.py
run_encoding_models_preference.py
run_encoding_models_residuals.py 
run_encoding_models_variance_partitioning.py
run_encoding_models_residuals.py
run_encoding_models.py 
run_fsl_randomize.py 
run_func_jobs.sh
run_func_nn_jobs.py
run_plot_jobs.py
run_sl_jobs.py 
run_srm_jobs.py 
save_full_joint_layer_reps.py
save_full_layer_plots.py 
save_parcel_data.py 
silent_features.npy 
sklearn_oldversion.txt
Sound_Envelope.ipynb
space_job.sh
Special_Visualizations.ipynb
Stuff_for_Sam.ipynb
test_loading.py
test_make_feats.py
test_optimization.py 
View_Band_Results.ipynb
View_Comparison_Maps.ipynb
View_Encoding_Visualizations.ipynb
View_GPTXL_Results_Raw.ipynb
View_Head_Only_Analyses.ipynb
View_Headwise_Results.ipynb
View_Normalized_ROI_Matrices.ipynb
View_Regularization_Coeffs.ipynb
View_ROI_Matrices.ipynb
View_ROI_Weigts.ipynb
View_RSA_Results.ipynb
View_Silent_Results.ipynb
banded_ridge_regression_joint.py
