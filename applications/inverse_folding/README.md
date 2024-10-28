# Pre-trained weights
To use our pre-trained weights, download the `inverse_folding.tar.gz` file from zenodo and place the unpacked `pretrained_weights folder in this directory

# Training Flow-Matching Inverse Folding Model (FMIF)
To train the flow-matching inverse folding  model, first, download the data used to train [ProteinMPNN](https://github.com/dauparas/ProteinMPNN/tree/main). The files can be downloaded from [here](https://github.com/dauparas/ProteinMPNN/tree/main)

FMIF can then be trained by running:
`python utils/train_flow_model.py --backbone_noise 0.1 --label_smoothing 0.0909 --path_for_training_data /YOUR/PATH/TO/PROTEIN/MPNN/DATA/pdb_2021aug02 --path_for_outputs /YOUR/PATH/TO/OUTPUT/WEIGHTS`

An example slurm script is also provided in `./slurm_scripts/launch_fmif_training.sh`

Pretrained weights are available in `./pretrained_weights/fmif_weights.pt`

# Setup Stability Data
From https://zenodo.org/records/7992926 download `AlphaFold_model_PDBs.zip` and place the unzipped directory inside `./utils/rocklin_data`
Run `process_name_to_graph.py` to pre-process the PDB files into the proper format for ProteinMPNN/FMIF. This should produce the file `name_to_graph.pt` in the directory

From https://zenodo.org/records/7992926 download `Processed_K50_dG_datasets.zip` and place the file `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` inside `./utils/rocklin_data`


# Train the clean (unnoised) stability regression model

`python utils/train_full_stability_model.py --model_path ./pretrained_weights/fmif_weights.pt`

# Train the noisy DDG>0 classifiers
This is a two step process. First we train an initial noisy classifier using just labeled data:

For example, to train a classifier for the protein in Rocklin cluster 7:
`python train_noisy_classifier.py --cluster 7` 
Move the results file, `noisy_classifier_7_30.pt` to `./pretrained_weights`

Then we sample more sequences with FMIF and label them with the regression model
`python train_noisy_classifier_iterative.py --cluster 7` 

The file, `noisy_classifier_7_iter_1.pt` is our final noisy classifier. Place this in `./pretrained_ewights`

# Run Guidance
To perform guidance using guide temperatures of 1.0, 0.1, 0.01 (guidance strengths of 1.0, 10, 100) for this cluster we run:

`python generate_sequences.py --guide_temp 1.0 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`
`python generate_sequences.py --guide_temp 0.1 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`
`python generate_sequences.py --guide_temp 0.01 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`

To perform TAG guidance using guide temperatures of 1.0, 0.1, 0.01 (guidance strengths of 1.0, 10, 100) for this cluster we run:

`python generate_sequences.py --use-tag --guide_temp 1.0 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`
`python generate_sequences.py --use-tag --guide_temp 0.1 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`
`python generate_sequences.py --use-tag --guide_temp 0.01 --batch_size 1 --x1_temp 0.1 --cluster '7'  --predictor_weights './pretrained_weights/noisy_classifier_7_iter_1.pt' --dt 0.01`

