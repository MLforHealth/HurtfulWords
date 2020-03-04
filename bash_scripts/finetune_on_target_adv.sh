#!/bin/bash
#SBATCH --partition p100 
#SBATCH --gres gpu:2
#SBATCH -c 8
#SBATCH --output=finetune_%A.out
#SBATCH --mem 40gb

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all, outhosp_mort}
# $2 - BERT model name {baseline_clinical_BERT_1_epoch_512, adv_clinical_BERT_1_epoch_512}
# $3 - dev fold
# $4, $5 -test folds
# $6 - lambda

set -e 
source activate hurtfulwords

BASE_DIR="/h/haoran/projects/HurtfulWords"
OUTPUT_DIR="/h/haoran/projects/HurtfulWords/data"
mkdir -p "$OUTPUT_DIR/models/finetuned/"

cd "$BASE_DIR/scripts"

python finetune_on_target_multitask.py \
    --df_path "${OUTPUT_DIR}/finetuning/$1" \
    --target_type "$1" \
    --model_path "${OUTPUT_DIR}/models/$2" \
    --output_dir "${OUTPUT_DIR}/models/finetuned/debiasing_finetuning_${6}/${2}/${1}/${3}/" \
    --train_batch_size 16 \
    --other_fields age sofa sapsii_prob sapsii_prob oasis oasis_prob \
    --dev_folds "$3" \
    --test_folds "$4" "$5" \
    --seed "$3" \
    --use_adversary \
    --lm "$6" \
    --protected_group gender \
    --adv_layers 3 \
    --max_num_epochs 5
