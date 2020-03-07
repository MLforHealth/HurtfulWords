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
import hashlib
import warnings
warnings.filterwarnings('ignore') 
hex_characters = '0123456789abcdef'

parser = argparse.ArgumentParser('''Goes through the folder where finetuned models are stored, outputs an excel file to each model folder,
                                    with various performance and fairness metrics.''')
parser.add_argument("--models_path",help = 'Root folder where finetuned models are stored. Each model should consist of several folders, each representing a fold', type=str)
parser.add_argument('--set_to_use', default = 'test', const = 'test', nargs = '?', choices = ['val', 'test'])
parser.add_argument('--overwrite', help = 'whether or not to overwrite existing excel files, or ignore them', action = 'store_true')
parser.add_argument('--bootstrap', action = 'store_true', help = 'whether to use bootstrap')
parser.add_argument('--bootstrap_samples', type = int, default = 1000, help = 'how many samples to bootstrap')
parser.add_argument('--save_folds', action = 'store_true', help = 'whether to output folds in excel file')
parser.add_argument('--task_split', default = 'all', const='all', nargs = '?', choices = list(hex_characters) + ['all'] )
args = parser.parse_args()

protected_groups = ['insurance', 'gender', 'ethnicity_to_use', 'language_to_use']
Constants.drop_groups['insurance'] += ['Government']
set_to_use = args.set_to_use
mapping = Constants.mapping

def compute_opt_thres(target, pred):
    opt_thres = 0
    opt_f1 = 0
    for i in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(target, pred >= i)
        if f1 >= opt_f1:
            opt_thres = i
            opt_f1 = f1
    return opt_thres

def stratified_sample(df, column, N):
    grp = df.groupby(column, group_keys = False)
    return grp.apply(lambda x: x.sample(n = int(np.rint(N*len(x)/len(df))), replace = True)).sample(frac=1).reset_index(drop = True)

def read_pickle_preds(df, merged_preds, key):
    temp = pd.DataFrame.from_dict(merged_preds, orient = 'index').reset_index().rename(columns = {'index': 'note_id', 0: 'pred'})
    temp = pd.merge(temp, df[['note_id','fold']], on = 'note_id', how = 'left')

    def fold_transform(x):
        if x == 'test': return x
        elif x in key['fold_id']: return 'val'
        else: return 'train'
    temp['fold'] = temp['fold'].apply(fold_transform)
    return temp

def compute_p_from_bootstrap(values):
    values = np.array(values)
    values = values[~pd.isnull(values)]
    pos_p = (values <= 0).sum()/len(values) 
    neg_p = (values >= 0).sum()/len(values)
    # choose hypothesis that gap is <> 0 based on whichever one is closer
    return min([neg_p, pos_p]), 1 if pos_p < neg_p else -1

def analyze_results(path):
    outfile_name = os.path.join(path, 'results.xlsx')
    key = json.load(open(os.path.join(path, 'argparse_args.json'), 'r'))
    df = pd.read_pickle(key['df_path'])
    target = key['target_col_name']
    task_type = key['task_type']

    if 'note_id' not in df.columns:
        df = df.reset_index()
    preds = read_pickle_preds(df, pickle.load(open(os.path.join(key['output_dir'], 'preds.pkl'), 'rb')), key)

    if target in ('insurance_enc', 'gender_enc', 'ethnicity_to_use_enc', 'language_to_use_enc'):
        prop_name = re.findall(r'(.*)_enc', target)[0]
        labels = []
        for i in np.sort(np.unique(df[target])):
            for idx,m in mapping[prop_name].items():
                if m == i:
                    labels.append(idx)
                    break
    else:
        prop_name = None
        labels = None

    cols_in_output = [k for k in ['all'] + protected_groups if k != prop_name]
    result_dfs = {i: pd.DataFrame(columns = ['avg']) for i in cols_in_output}

    temp = pd.merge(preds, df[['note_id',target]+ protected_groups], on = 'note_id', how = 'left')
    val = temp[temp['fold'] == 'val']
    thres = compute_opt_thres(val[target], val['pred'])
    temp = temp[temp['fold'] == set_to_use]

    if args.bootstrap:
        for s in tqdm(range(1, args.bootstrap_samples+1)):
            df_sample = stratified_sample(temp, target, len(temp))
            calc_metrics(result_dfs, cols_in_output, df_sample, labels, prop_name, s, task_type, target, df, thres)
    else:
        calc_metrics(result_dfs, cols_in_output, temp, labels, prop_name, 1, task_type, target, df, thres)

    for key, result_df in result_dfs.items():
        if args.bootstrap:
            for idx, row in result_df.iterrows():
                values = [row['fold_%s'%i] for i in range(1, args.bootstrap_samples+1)]
                result_df.loc[idx, 'avg'] = np.nanmean(values)
                result_df.loc[idx, 'std'] = np.nanstd(values, ddof = 1)
                errors = np.nanmean(values) - values
                result_df.loc[idx, '2.5%'] = result_df.loc[idx, 'avg'] - np.nanpercentile(errors, 97.5)
                result_df.loc[idx, '97.5%'] = result_df.loc[idx, 'avg'] - np.nanpercentile(errors, 2.5)
                if 'gap' in idx:
                    p, direction = compute_p_from_bootstrap(values)
                    result_df.loc[idx, 'favor'] = direction
                    result_df.loc[idx, 'p'] = p
        else:
            for idx, row in result_df.iterrows():
                values = [row['fold_1']]
                result_df.loc[idx, 'avg'] = np.nanmean(values)

    # add threshold as separate sheet
    result_dfs['thres'] = pd.DataFrame([thres])

    with pd.ExcelWriter(outfile_name) as writer:
        for i in result_dfs:
            if args.bootstrap and i !='thres':
                if args.save_folds:
                    result_dfs[i].to_excel(writer, sheet_name = i)
                else:
                    if i == 'all':
                        result_dfs[i][['avg','2.5%','97.5%','std']].to_excel(writer, sheet_name = i)
                    else:
                        result_dfs[i][['avg','2.5%','97.5%','std', 'favor', 'p']].to_excel(writer, sheet_name = i)
            else:
                result_dfs[i].to_excel(writer, sheet_name = i)

def calc_metrics(result_dfs, cols_in_output, df_fold, labels, prop_name, c, task_type, target, df, thres):
    for g in cols_in_output:
        if g != 'all':
            df_fold = df_fold[~df_fold[g].isin(Constants.drop_groups[g])]
            refined_mapping = {i:j for i,j in mapping[g].items() if i not in Constants.drop_groups[g]}

        if task_type == 'binary':
            if g == 'all':
                calc_binary(df_fold, result_dfs[g], c, 'all', target, thres, None, prop_name, labels)
            else:
                for j in refined_mapping:
                    q = '%s=="%s"'%(g, j)
                    calc_binary(df_fold.query(q), result_dfs[g], c, q,target, thres, result_dfs['all'],prop_name, labels)

                for a,b in {'pred_prevalence': 'dgap', 'recall': 'egap_positive', 'specificity': 'egap_negative'}.items(): #computes gap_max for each group
                    df_fold_gap = result_dfs[g].loc[result_dfs[g].index.str.endswith(a), 'fold_%s'%c]
                    for j in refined_mapping:
                        q = '%s=="%s"'%(g, j)
                        curnum = df_fold_gap[df_fold_gap.index.str.startswith(q)].iloc[0]
                        diffs = [curnum - i for i in df_fold_gap[~df_fold_gap.index.str.startswith(q)]]
                        maxDiffIdx = np.abs(diffs).argmax()
                        result_dfs[g].loc['%s_%s_max'%(q,b),'fold_%s'%c] = diffs[maxDiffIdx]
        else:
            raise Exception("Invalid task type!")


def calc_binary(temp, result_df, fold_id, prefix, target, thres, all_df = None, prop_name = None, labels = None):
    metrics = {}
    if temp.shape[0] == 0:
        return metrics
    if len(np.unique(temp[target])) > 1:
        metrics['auroc'] = roc_auc_score(temp[target], temp['pred'])
    metrics['precision'] = precision_score(temp[target], temp['pred'] >= thres)
    metrics['recall'] = recall_score(temp[target], temp['pred'] >= thres)
    metrics['auprc'] = average_precision_score(temp[target], temp['pred'])
    metrics['log_loss'] = log_loss(temp[target], temp['pred'], labels = [0, 1])
    metrics['acc'] = accuracy_score(temp[target], temp['pred'] >= thres)
    CM = confusion_matrix(temp[target], temp['pred'] >= thres, labels = [0, 1])
    metrics['TN'] = CM[0][0]
    metrics['FN'] = CM[1][0]
    metrics['TP'] = CM[1][1]
    metrics['FP'] = CM[0][1]
    metrics['class_true_count'] = (temp[target] == 1).sum()
    metrics['class_false_count']= (temp[target] == 0).sum()
    metrics['specificity'] = float(CM[0][0])/(CM[0][0] + CM[0][1]) if metrics['class_false_count'] > 0 else 0
    metrics['pred_true_count'] = ((temp['pred'] >= thres) == 1).sum()
    metrics['nsamples'] = len(temp)
    metrics['pred_prevalence']= metrics['pred_true_count'] /float(len(temp))
    metrics['actual_prevalence'] = metrics['class_true_count']/ float(len(temp))

    for i,m in metrics.items():
        result_df.loc['%s_%s'%(prefix,i), 'fold_%s'%fold_id] = m

def hash(x):
    return hashlib.md5(x.encode()).hexdigest()

for folder in os.scandir(args.models_path):
    if (folder.is_dir()) and ((not any([filename.endswith('.xlsx') for filename in os.listdir(folder.path)])) or args.overwrite):
        if os.path.exists(os.path.join(folder.path, 'rough_preds.pkl')):
            os.rename(os.path.join(folder.path, 'rough_preds.pkl'), os.path.join(folder.path, 'preds.pkl'))
        if os.path.exists(os.path.join(folder.path, 'preds.pkl')):
            if args.task_split == 'all' or (hash(folder.path)[0] == args.task_split):
                print('Starting %s' % folder.path)
                analyze_results(folder.path)
                print('Finished %s' % folder.path)
        else:
            print('Skipping incomplete %s' % folder.path)
