# Change these locations as necessary
set -e 
source activate hurtfulwords

BASE_DIR="/h/haoran/projects/HurtfulWords/" 
OUTPUT_DIR="/h/haoran/projects/HurtfulWords/data/"

cd "$BASE_DIR/scripts"

python log_probability_bias_scores.py \
    --model "${OUTPUT_DIR}/models/baseline_clinical_BERT_1_epoch_512/" \
    --demographic 'GEND' \
    --attribute 'DRUG' \
    --template_file "${BASE_DIR}/fill_in_blanks_examples/templates.txt" \
    --attributes_file "${BASE_DIR}/fill_in_blanks_examples/attributes.csv" \
    --out_file "${OUTPUT_DIR}/log_probability_bias_predictions.tsv" 
 
python statistical_significance.py "${OUTPUT_DIR}/log_probability_bias_predictions.tsv" > "${OUTPUT_DIR}/log_probability_statistical_significance.txt"

