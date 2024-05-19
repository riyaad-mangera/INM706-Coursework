Link to GitHub repository: https://github.com/riyaad-mangera/INM706-Coursework
Link to dataset and checkpoint: https://drive.google.com/drive/folders/1VEQDG_znWXsx0Dm9NF7iSy5a1RKm3VnE

Please place these two folders in the root directory.

# How To Run
Python Version: 3.9.5

## City Hyperion [linux server]
```
sh create_env.sh
```

## Windows:
```
.\setup.ps1
```

Sometimes the scripts will not properly install all required libraries and a manual pip install is required for said libraries.
Before manual installing please attempt to run these files a seconf time 

# Checkpoints
Final Model checkpoints are included in the repo under the trained_models folder.

# Training

Parameters controlable from config.yaml are:
 - Number of epochs
 - Batch size
 - Learning rate
 - Rate of decay
 - Embedding dimensions
 - Didden dimensions
 - Percent of train set to use
 - Percent of test set to use
 - Percent of valid set to use

The wand api key should be added to the run_job.sh on linux or entered as requested by Windows.

During training, you will see the most recent epoch in a folder called checkpoints, and the final model will generate a pkl file in the trained models folder.

# Logging

The Wandb logs for training can be found here:

 - Original Dataset Logs
    - https://wandb.ai/riyaad-mangera/inm706_cw

- ontonotes5 Dataset Logs
    - https://wandb.ai/riyaad-mangera/inm706_cw_simpler_dataset
    - https://wandb.ai/riyaad-mangera/inm706_cw_hyperion_runs_simpler_dataset


# Inference

There is an inference for each model in the inference.ipynb file, results have been preselected to provide a sample output.
