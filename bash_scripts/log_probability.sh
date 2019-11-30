# Change these locations as necessary
OUT_FILE="/h/amyxlu/HurtfulWords/fill_in_blanks_examples/results/log_probability_bias_predictions.tsv"
MODEL_DIR='/scratch/gobi1/haoran/shared_data/BERT_DeBias/models/baseline_clinical_BERT_1_epoch_512'
BASE_DIR="/h/amyxlu/HurtfulWords"

cd "$BASE_DIR/scripts"

python log_probability_bias_scores.py \
    --model $MODEL_DIR \
    --demographic 'GEND' \
    --attribute 'DRUG' \
    --template_file "${BASE_DIR}/fill_in_blanks_examples/templates.txt" \
    --attributes_file "${BASE_DIR}/fill_in_blanks_examples/attributes.csv" \
    --out_file $OUT_FILE 
 
python statistical_significance.py $OUT_FILE > ${BASE_DIR}/fill_in_blanks_examples/results/statistical_significance.txt 

cd $BASE_DIR/bash_scripts 
