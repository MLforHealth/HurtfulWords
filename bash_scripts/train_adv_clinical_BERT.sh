#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:8
#SBATCH -c 8
#SBATCH --output train_adv%A.log
#SBATCH --mem 160gb
set -e
source activate hurtfulwords

BASE_DIR="/h/haoran/projects/HurtfulWords"
OUTPUT_DIR="/scratch/hdd001/home/haoran/shared_data/BERT_DeBias/data/"
SCIBERT_DIR="/scratch/hdd001/home/haoran/shared_data/BERT_DeBias/models/SciBERT"
mkdir -p "$OUTPUT_DIR/models/"
DOMAIN="$1"

cd "$BASE_DIR/scripts" 

python adversarial_finetune_on_pregen.py \
	--pregenerated_data "$OUTPUT_DIR/pregen_epochs/128/" \
	--output_dir "$OUTPUT_DIR/models/adv_clinical_BERT_${DOMAIN}_1_epoch_128/" \
	--bert_model "$SCIBERT_DIR" \
	--do_lower_case \
	--epochs 1 \
	--train_batch_size 64\
	--seed 123 \
	--domain_of_interest "$DOMAIN" \
	--lambda_ 1.0 \
	--num_layers 3\
    --use_new_mapping

python adversarial_finetune_on_pregen.py \
	--pregenerated_data "$OUTPUT_DIR/pregen_epochs/512/" \
	--output_dir "$OUTPUT_DIR/models/adv_clinical_BERT_${DOMAIN}_1_epoch_512/" \
	--bert_model "$OUTPUT_DIR/models/adv_clinical_BERT_${DOMAIN}_1_epoch_128/" \
	--do_lower_case \
	--epochs 1 \
	--train_batch_size 32\
	--seed 123 \
	--domain_of_interest "$DOMAIN" \
	--lambda_ 1.0 \
	--num_layers 3\
    --use_new_mapping


rm -rf "$OUTPUT_DIR/models/adv_clinical_BERT_${DOMAIN}_1_epoch_128/"
