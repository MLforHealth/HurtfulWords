#!/bin/sh
#SBATCH --partition cpu 
#SBATCH -c 2
#SBATCH --output bootstrap%A.log
#SBATCH --mem 50gb

set -e
source activate hurtfulwords

BASE_DIR="/h/haoran/projects/HurtfulWords"
OUTPUT_DIR="/h/haoran/projects/HurtfulWords/data/"
cd "$BASE_DIR/scripts"

python analyze_results.py \
	--models_path "${OUTPUT_DIR}/models/finetuned/" \
	--set_to_use "test" \
	--bootstrap \
