#!/h/haoran/anaconda3/bin/python
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import argparse
import Constants
import json
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score,\
    log_loss, precision_score, confusion_matrix, recall_score, f1_score
import re
from tqdm import tqdm
import pickle
hex_characters = '0123456789abcdef'

parser = argparse.ArgumentParser('''Goes through the folder where finetuned models are stored, outputs an excel file to each model folder,
                                    with various performance and fairness metrics.''')
parser.add_argument("--models_path",help = 'Root folder where finetuned models are stored. Each model should consist of several folders, each representing a fold', type=str)
parser.add_argument('--overwrite', help = 'whether or not to overwrite existing excel files, or ignore them', action = 'store_true')
parser.add_argument('--save_folds', action = 'store_true', help = 'whether to output folds in excel file')
args = parser.parse_args()

protected_groups = ['insurance', 'gender', 'ethnicity_to_use', 'language_to_use']
Constants.drop_groups['insurance'] += ['Government']
mapping = Constants.mapping

def read_pickle_preds(df, merged_preds, key, targets):
    temp = pd.DataFrame.from_dict(merged_preds, orient = 'index', columns = ['pred' + i for i in targets]).reset_index().rename(columns = {'index': 'note_id'})
    temp = pd.merge(temp, df[['note_id','fold']], on = 'note_id', how = 'left')

    def fold_transform(x):
        if x in ['test', *key['test_folds']]: return 'test'
        elif x in ['val', *key['dev_folds']]: return 'val'
        else: return 'train'
    temp['fold'] = temp['fold'].apply(fold_transform)
    return temp

def compute_opt_thres(target, pred):
    opt_thres = 0
    opt_f1 = 0
    for i in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(target, pred >= i)
        if f1 >= opt_f1:
            opt_thres = i
            opt_f1 = f1
    return opt_thres

def analyze_results(path):
    outfile_name = os.path.join(path, 'results.xlsx')
    key = json.load(open(os.path.join(path, 'argparse_args.json'), 'r'))
    targets = Constants.targets[key['target_type']]
    df = pd.read_pickle(key['df_path'])

    if 'note_id' not in df.columns:
        df = df.reset_index()
    preds = read_pickle_preds(df, pickle.load(open(os.path.join(key['output_dir'], 'preds.pkl'), 'rb')), key, targets)

    cols_in_output = [k for k in ['all'] + protected_groups]
    result_dfs = {i: pd.DataFrame(columns = [], index = targets) for i in cols_in_output}

    temp = pd.merge(preds, df[['note_id',*targets]+ protected_groups], on = 'note_id', how = 'left')

    thresholds = {}
    val_df = temp[temp.fold == 'val']
    test_df = temp[temp.fold == 'test']
    for t in targets:
        thresholds[t] = compute_opt_thres(val_df[t], val_df['pred' + t])

    result_dfs['all']['threshold'] = pd.Series(thresholds)

    calc_metrics(result_dfs, cols_in_output, temp, targets, df, thresholds)

    with pd.ExcelWriter(outfile_name) as writer:
        for i in result_dfs:
            result_dfs[i].to_excel(writer, sheet_name = i)

def calc_metrics(result_dfs, cols_in_output, df_fold, targets, df, thresholds):
    for g in cols_in_output:
        if g != 'all':
            df_fold = df_fold[~df_fold[g].isin(Constants.drop_groups[g])]
            refined_mapping = {i:j for i,j in mapping[g].items() if i not in Constants.drop_groups[g]}

        if g == 'all':
            calc_binary(df_fold, result_dfs[g], 'all', targets, thresholds)
        else:
            for j in refined_mapping:
                q = '%s=="%s"'%(g, j)
                calc_binary(df_fold.query(q), result_dfs[g], q,targets, thresholds)

            for t in targets:
                for a,b in {'pred_prevalence': 'dgap', 'recall': 'egap_positive', 'specificity': 'egap_negative'}.items(): #computes gap_max for each group
                    df_fold_gap = result_dfs[g].T.loc[result_dfs[g].T.index.str.endswith(a), t]
                    for j in refined_mapping:
                        q = '%s=="%s"'%(g, j)
                        curnum = df_fold_gap[df_fold_gap.index.str.startswith(q)].iloc[0]
                        diffs = [curnum - i for i in df_fold_gap[~df_fold_gap.index.str.startswith(q)]]
                        maxDiffIdx = np.abs(diffs).argmax()
                        result_dfs[g].loc[t, '%s_%s_max'%(q,b)] = diffs[maxDiffIdx]


def calc_binary(temp, result_df, prefix, targets, thresholds):
    if temp.shape[0] == 0:
        return None
    for t in targets:
        thres = thresholds[t]
        metrics = {}
        pred_col = 'pred' + t
        if len(np.unique(temp[t])) > 1:
            metrics['auroc'] = roc_auc_score(temp[t], temp[pred_col])
        metrics['precision'] = precision_score(temp[t], temp[pred_col] >= thres)
        metrics['recall'] = recall_score(temp[t], temp[pred_col] >= thres)
        metrics['auprc'] = average_precision_score(temp[t], temp[pred_col])
        metrics['log_loss'] = log_loss(temp[t], temp[pred_col], labels = [0, 1])
        metrics['acc'] = accuracy_score(temp[t], temp[pred_col] >= thres)
        CM = confusion_matrix(temp[t], temp[pred_col] >= thres, labels = [0, 1])
        metrics['TN'] = CM[0][0]
        metrics['FN'] = CM[1][0]
        metrics['TP'] = CM[1][1]
        metrics['FP'] = CM[0][1]
        metrics['class_true_count'] = (temp[t] == 1).sum()
        metrics['class_false_count']= (temp[t] == 0).sum()
        metrics['specificity'] = float(CM[0][0])/(CM[0][0] + CM[0][1]) if metrics['class_false_count'] > 0 else 0
        metrics['pred_true_count'] = ((temp[pred_col] >= thres) == 1).sum()
        metrics['nsamples'] = len(temp)
        metrics['pred_prevalence']= metrics['pred_true_count'] /float(len(temp))
        metrics['actual_prevalence'] = metrics['class_true_count']/ float(len(temp))

        for i,m in metrics.items():
            result_df.loc[t, '%s_%s'%(prefix,i)] = m


for root, dirs, files in os.walk(args.models_path):
    if len(dirs) == 5 and dirs[0].isdigit():
        for split in dirs:
            base_path = os.path.join(root, split)
            if 'preds.pkl' not in os.listdir(base_path) or ((any([filename.endswith('.xlsx') for filename in files])) and not args.overwrite):
                break
            print("Starting %s" % base_path)
            analyze_results(base_path)
