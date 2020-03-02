import os
import sys
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target_type", nargs = '?', choices = ['phenotype_all', 'phenotype_first', 'inhosp_mort', 'outhosp_mort'], type=str)
args = parser.parse_args()
std_models = ['baseline_clinical_BERT_1_epoch_512', 'adv_clinical_BERT_1_epoch_512']

folds = [(8, 9, 10),(10, 1, 2),(2,3, 4),(4,5,6),(6,7,8)]

for model in std_models:
    for f in folds:
        dev = str(f[0])
        test1,test2 = str(f[1]), str(f[2])      
        subprocess.call(shlex.split('sbatch finetune_on_target.sh "%s" "%s" "%s" "%s" "%s"'%(args.target_type, model, dev, test1, test2)))