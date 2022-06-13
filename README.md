## Reconstructing the cascade of language processing in the brain using the internal computations of transformer language models

![alt text](https://github.com/tsumers/bert-brains/blob/master/figure_1_github.png?raw=true)

This repository accompanies the manuscript "Reconstructing the cascade of language processing in the brain using the internal computations of transformer language models". In this manuscript we use the internal computations of a transformer model (BERT) to predict fMRI activity while subjects listen to naturalistic spoken narratives.

Kumar, S.\*, Sumers, T. R.\*, Yamakoshi, T., Goldstein, A., Hasson, U., Norman, K. A., Griffiths, T. L., Hawkins, R. D., & Nastase, S. A. (2022). Reconstructing the cascade of language processing in the brain using the internal computations of a transformer-based language model. *bioRxiv*. https://doi.org/10.1101/2022.06.08.495348

The following list describes the various analysis scripts:

### Text Processing and Transformer Models
`narratives-transcript-processing.ipynb`: Used to pre-process transcripts from the `Narratives` dataset. Outputs (1) phonemes and associated nuisance variables for regression analyses; (2) TR-aligned tokens for use with Transformer notebooks.

`transformer-representations.ipynb`: the primary notebook used to generate Transformer representations (embeddings, transformations) for regression analyses.  

`transformer-transformation-magnitudes.ipynb`: small script used to produce transformation magnitudes from the transformations themselves.  

`transformer-utils.py`: functionality for extracting various Transformer representations, including some experimental metrics that were used in the paper.  

`extract_linguistic_features.py`: Extract linguistic features (parts-of-speech and dependency tags) using spaCy.

`decode_linguistic_features.py`: Run the decoding analysis, where we decode linguistic features from the headwise transformation representations. All the necessary functions should be in decode_linguistic_features_utils.py.

### Data handling
`create_fmri_dataset.py`: Copy data from narratives dataset into lab project folder.

`ROIs.ipynb`: Handling ROI labeling and visualizing.


### Analysis code
`banded_ridge_regression.py`: Run encoding model analyses (banded ridge regression) with a given representation as main features.

`calculate_boostrap_pvalue.py`: Calculate bootstrap pvalue (with FDR corrections) for individual parcels to determine significance.

`compute_isc.py`: Compute noise ceilings with intersubject correlation.

`headwise_banded_ridge_regression.py`: Headwise version of the encoding analyses (banded ridge regression, but knock out weights for all heads but one when evaluating a head).

`run_encoding_models_banded.py`: Creates a joblist.txt for banded ridge regression jobs to input as a slurm job array.

`run_encoding_models_headwise_banded.py`: Same as above, but for the headwise encoding analyses.

`run_isc_jobs.py`: Run multiple subjects' ISC analyses in parallel (slurm jobs).

`run_jupyter.sh`: Run a jupyter notebook on the cluster.

`run_pvalue_jobs.py`: Run bootstrap p-value analyses in parallel.

`run_save_jobs.py`: Run jobs in parallel that will save collapsed performance across subjects.

`save_mean_volume.py`: Collapse performance across subjects for a specific representation.


### Visualizations
`Head_Dep_Brain_Plots.ipynb`: Plots whole brain results for headwise encoding (and dependency encoding).

`Layerwise_Plot.ipynb`: Plots performance across layers.

`Results_Figures.ipynb`: Notebook to plot most main results in the paper.

`View_Encoding_Results.ipynb`: Plots whole-brain glass brain results.

`bert_rdms.ipynb`: Compare RDMs for embeddings and transformations (as well as autocorrelation).

`layer_preference.ipynb`: Create layer preference and layer specificity histograms.

`plot_layer_brain.ipynb`: Process and export layer preferences for visualization on cortical surface.

`headwise_specialization.ipynb`: Run PCA on transformation weights and visualization two-dimensional projections.


### Slurm jobs
`avg_job.sh`: Slurm job corresponding to save_mean_volumes.py.

`boot_job.sh`: Slurm job corresponding to calculate_bootstrap_pvalue.py.

`dsq-submit.sh`: Slurm job that submits a job array specified in a .txt file.

`joblist.txt`: Job list for submitting slurm job array.


### Miscellaneous
`fmri_conda_env.yml`: yml conda env file for fmri analyses.

`transformer_conda_env.yml`: yml conda env file for transcript pre-processing and Transformer analysis notebooks.
