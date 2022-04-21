# bert-brains

### Preprocessing
`narratives-transcript-processing.ipynb`: Used to pre-process transcripts from the `Narratives` dataset. Outputs (1) phonemes and associated nuisance variables for regression analyses; (2) TR-aligned tokens for use with Transformer notebooks.

### Transformers
`transformer-representations.ipynb`: the primary notebook used to generate Transformer representations (embeddings, transformations) for regression analyses.  

`transformer-transformation-magnitudes.ipynb`: small script used to produce transformation magnitudes from the transformations themselves.  

`transformer-utils.py`: functionality for extracting various Transformer representations, including some experimental metrics that were used in the paper.  

### data_moving/handling
`create_fmri_dataset.py`: Copy data from narratives dataset into lab project folder.

`ROIs.ipynb`: Handling ROI labeling and visualizing.

### Analysis Code
`banded_ridge_regression.py`: Run encoding model analyses (banded ridge regression) with a given representation as main features

`calculate_boostrap_pvalue.py`: Calculate bootstrap pvalue (with FDR corrections) for individual parcels to determine significance.

`compute_isc.py`: Compute noise ceilings with intersubject correlation.

`extract_linguistic_features.py`: Extract linguistic features (parts-of-speech and dependency tags) using spaCy.

`decode_linguistic_features.py`: Run the decoding analysis, where we decode linguistic features from the headwise transformation representations. All the necessary functions should be in decode_linguistic_features_utils.py.

`headwise_banded_ridge_regression.py`: Headwise version of the encoding analyses (banded ridge regression, but knock out weights for all heads but one when evaluating a head)

`run_encoding_models_banded.py`: Creates a joblist.txt for banded ridge regression jobs to input as a slurm job array.

`run_encoding_models_headwise_banded.py`: Same as above, but for the headwise encoding analyses.

`run_isc_jobs.py`: Run multiple subjects' ISC analyses in parallel (slurm jobs)

`run_jupyter.sh`: Run a jupyter notebook on the cluster.

`run_pvalue_jobs.py`: Run bootstrap p-value analyses in parallel.

`run_save_jobs.py`: Run jobs in parallel that will save collapsed performance across subjects.

`save_mean_volume.py`: Collapse performance across subjects for a specific representation.



### Visualizations
`Head_Dep_Brain_Plots.ipynb`: Plots whole brain results for headwise encoding (and dependency encoding).

`Layerwise_Plot.ipynb`: Plots performance across layers.

`Results_Figures.ipynb`: Notebook to plot most main results in the paper.

`View_Encoding_Results.ipynb`: Plots whole-brain glass brain results

### Slurm Jobs
`avg_job.sh`: Slurm job corresponding to save_mean_volumes.py.

`boot_job.sh`: Slurm job corresponding to calculate_bootstrap_pvalue.py

`dsq-submit.sh`: Slurm job that submits a job array specified in a .txt file.

`joblist.txt`: Job list for submitting slurm job array


### Misc
`fmri_conda_env.yml`: yml conda env file for fmri analyses  
`transformer_conda_env.yml`: yml conda env file for transcript pre-processing and Transformer analysis notebooks
