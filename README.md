# Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings

## Paper
If you use this code in your research, please cite the following publication:

## Step 0: Environment and Prerequisites
- Before starting, go to the [MIMIC-benchmarks repository](https://github.com/YerevaNN/mimic3-benchmarks), and follow all of the steps in the `Building a benchmark` section.
- Run the following commands to clone this repo and create the Conda environment
```
git clone https://github.com/MLforHealth/HurtfulWords.git
cd HurtfulWords/
conda create -y -n hurtfulwords python=3.7
conda activate hurtfulwords
pip install -r requirements.txt
```

## Step 1: Data processing
Reads in the tables from MIMIC and pregenerates data for clinical BERT pretraining. Reads in the cohorts defined by MIMIC-benchmarks and creates tasks for finetuning on downstream targets.
- In `bash_scripts/data_processing_pipeline.sh`, update `BASE_DIR`, `OUTPUT_DIR`, `SCIBERT_DIR` and `MIMIC_BENCHMARK_DIR`.
- In `scripts/get_data.py`, update the database connection credentials on line 13. If your MIMIC-III is not loaded into a database, you will have to update this script accordingly.
- Run `bash_scripts/data_processing_pipeline.sh`. This script will require at least 50 GB of RAM, 100 GB of disk space in `OUTPUT_DIR`, and will take several days to complete.

## Step 2: Training Baseline Clinical BERT
Pretrains baseline clinical BERT (initialized from SciBERT) for 1 epoch on sequences of length 128, then 1 epoch on sequences of length 512.
- In `bash_scripts/train_baseline_clinical_BERT.sh`, update `BASE_DIR`, `OUTPUT_DIR`, and `SCIBERT_DIR`. These variables should have the same values as in step 1.
- Run `bash_scripts/train_baseline_clinical_BERT.sh` on a GPU cluster. The resultant model will be saved in `$OUTPUT_DIR/models/baseline_clinical_BERT_1_epoch_512/`.

## Step 3: Training Adversarial Clinical BERT
Pretrains clinical BERT (initialized from SciBERT) with adversarial debiasing using gender as the protected attribute, for 1 epoch on sequences of length 128, then 1 epoch on sequences of length 512. 
- In `bash_scripts/train_adv_clinical_bert.sh`, update `BASE_DIR`, `OUTPUT_DIR`, and `SCIBERT_DIR`. These variables should have the same values as in step 1.
- Run `bash_scripts/train_adv_clinical_bert.sh` on a GPU cluster. The resultant model will be saved in `$OUTPUT_DIR/models/adv_clinical_BERT_1_epoch_512/`.


## Step 4: Finetuning on Downstream Tasks

## Step 5: Log Probabiltiy Bias Score
Following procedures in [Kurita et al.](http://arxiv.org/abs/1906.07337), we calculate the 'log probability bias score' to evaluate biases in the BERT model. 

## Step 6: Sentence Completion 

## Step 6: Sentence Completion
`scripts/predict_missing.py` takes template sentences which contain `_` for tokens to be predicted. It loops through a list of models to be asessed. Template sentences can be specified directly in the script.

To generate sentences, we loop through a list of variations for each demographic keyword, and see what the model will replace blank values with. 

An example of input templates and results can be found in `fill_in_blanks_examples/templates/` and `fill_in_blanks_examples/results/`, respectively.
