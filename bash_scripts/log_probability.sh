set -e 
source activate hurtfulwords

BASE_DIR="/h/haoran/projects/HurtfulWords/" 
OUTPUT_DIR="/h/haoran/projects/HurtfulWords/data/"
#MODEL_NAME="baseline_clinical_BERT_1_epoch_512"
#MODEL_NAME="adv_clinical_BERT_1_epoch_512" 
MODEL_NAME="SciBERT"

cd "$BASE_DIR/scripts"

python log_probability_bias_scores.py \
    --model "${OUTPUT_DIR}/models/${MODEL_NAME}/" \
    --demographic 'GEND' \
    --template_file "${BASE_DIR}/fill_in_blanks_examples/templates.txt" \
    --attributes_file "${BASE_DIR}/fill_in_blanks_examples/attributes.csv" \
    --out_file "${OUTPUT_DIR}/${MODEL_NAME}_log_scores.tsv" 
 
python statistical_significance.py "${OUTPUT_DIR}/${MODEL_NAME}_log_scores.tsv" > "${OUTPUT_DIR}/${MODEL_NAME}_log_score_significance.txt"

