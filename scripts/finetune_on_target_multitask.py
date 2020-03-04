import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import argparse
import Constants
import torch
import torch.nn as nn
from torch.utils import data
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
from run_classifier_dataset_utils import InputExample, convert_examples_to_features
from pathlib import Path
from tqdm.auto import tqdm
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from gradient_reversal import GradientReversal
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss, mean_squared_error, classification_report
import random
import json
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from utils import create_hdf_key, Classifier, get_emb_size, MIMICDataset, extract_embeddings, EarlyStopping, load_checkpoint
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser('Fine-tunes a pre-trained BERT model on a certain target for one fold. Outputs fine-tuned BERT model and classifier, ' +
                                 'as well as a pickled dictionary mapping id: predicted probability')
parser.add_argument("--df_path",help = 'must have the following columns: seqs, num_seqs, fold, with note_id as index', type=str)
parser.add_argument("--target_type", nargs = '?', choices = ['phenotype_all', 'phenotype_first', 'inhosp_mort', 'outhosp_mort'], type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--output_dir", help = 'folder to output model/results', type=str)
parser.add_argument('--train_batch_size', help = 'batch size to use for training', type = int)
parser.add_argument('--eval_batch_size', help = 'batch size to use for evaluation', type = int, default = 32)

parser.add_argument('--dev_folds', help = 'what fold to use as the DEV fold. Dataframe must have a "fold" column',nargs = '+', type=str, dest = 'dev_folds', default = [])
parser.add_argument('--test_folds', help = 'what fold to use as the TEST fold. Dataframe must have a "fold" column',nargs = '+', type=str, dest = 'test_folds', default = [])
parser.add_argument('--emb_method', default = 'cat4', const = 'cat4', nargs = '?', choices = ['last', 'sum4', 'cat4'], help = 'what embedding layer to take')

parser.add_argument('--use_adversary', help = "whether or not to use an adversary. If True, must not have --freeze_bert", action = 'store_true')
parser.add_argument('--lm', help = 'lambda value for the adversary', type = float, default = 1.0)
parser.add_argument('--protected_group', help = 'name of protected group, must be a column in the dataframe', type = str, default = 'gender')
parser.add_argument('--adv_layers', help = 'number of layers in adversary', type = int, default = 2)
parser.add_argument('--use_new_mapping', help = 'whether to use new mapping for adversarial training', action = 'store_true')

parser.add_argument('--max_num_epochs', help = 'maximum number of epochs to train for', type = int, default = 3)
parser.add_argument('--es_patience', help = 'patience for the early stopping', type = int, default = 3)
parser.add_argument('--other_fields', help = 'other fields to add, must be columns in df', nargs = '+', type = str, dest = 'other_fields', default = [])
parser.add_argument('--seed', type = int, default = 42, help = 'random seed for initialization')
parser.add_argument('--lr', type = float, default = 5e-3, help = 'learning rate for BertAdam optimizer')

args = parser.parse_args()
print(vars(args))

print('Reading dataframe...', flush = True)
df = pd.read_pickle(args.df_path)
if 'note_id' in df.columns:
    df = df.set_index('note_id')
    
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertModel.from_pretrained(args.model_path)
targets = Constants.targets[args.target_type]
for i in targets:
    assert(i in df.columns)
other_fields_to_include = args.other_fields

if args.use_adversary:
    protected_group = args.protected_group
    assert(protected_group in df.columns)
    if args.use_new_mapping:
        mapping = Constants.newmapping
        for i in Constants.drop_groups[protected_group]:
            df = df[df[protected_group] != i]
    else:
        mapping = Constants.mapping
        
for i in other_fields_to_include: # normalize features
    temp = df.loc[~df.fold.isin(['test', 'NA', *args.test_folds]), i]
    mean, stdev = np.mean(temp), np.std(temp)
    df[i] = (df[i] - mean)/stdev

c_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000]

Path(args.output_dir).mkdir(parents = True, exist_ok = True)

EMB_SIZE = get_emb_size(args.emb_method)
train_df = df[~df.fold.isin(['test', 'NA', *args.test_folds])]
val_df = df[df.fold.isin(args.dev_folds)]
test_df = df[(df.fold == 'test') | (df.fold.isin(args.test_folds))]

def convert_input_example(note_id, text, seqIdx, targets, group, other_fields = []):
    return InputExample(guid = '%s-%s'%(note_id,seqIdx), text_a = text, text_b = None, 
                        label = targets, group = mapping[protected_group][group] if args.use_adversary else 0, other_fields = other_fields)

print('Converting input examples to appropriate format...', flush = True)
examples_train = [convert_input_example(idx, i, c, row[targets].values.astype(np.float32), row[protected_group] if args.use_adversary else 0,
                                       [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in train_df.iterrows()
                  for c, i in enumerate(row.seqs)]

examples_eval = [convert_input_example(idx, i, c, row[targets].values.astype(np.float32), row[protected_group] if args.use_adversary else 0,
                                      [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in val_df.iterrows()
                  for c, i in enumerate(row.seqs)]

examples_test = [convert_input_example(idx, i, c, row[targets].values.astype(np.float32), row[protected_group] if args.use_adversary else 0,
                                      [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in test_df.iterrows()
                  for c, i in enumerate(row.seqs)]

class Discriminator(nn.Module): 
    def __init__(self, input_dim, num_layers, num_categories, lm):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        assert(num_layers >= 1)
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.lm = lm
        self.layers = [GradientReversal(lambda_ = lm)]
        for c, i in enumerate(range(num_layers)):
            if c != num_layers-1:
                self.layers.append(nn.Linear(input_dim // (2**c), input_dim // (2**(c+1))))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(input_dim // (2**c), num_categories))
                self.layers.append(nn.Softmax(dim = 0))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

predictor_params = {
    'num_layers': 1,
    'dropout_prob': 0,
     'decay_rate': 2, 
    'input_dim': EMB_SIZE + len(other_fields_to_include),
    'task_type': 'binary',
    'num_outputs' : len(targets)    
}

if args.use_adversary:
    discriminator = Discriminator(EMB_SIZE, args.adv_layers, len(mapping[protected_group]), args.lm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
assert(torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
device_ids = [i for i in range(torch.cuda.device_count())]
model = model.to(device)
if args.use_adversary:
    discriminator.to(device)
    
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

criterion = nn.BCELoss()
criterion_adv = nn.CrossEntropyLoss()
if n_gpu > 1:    
    model = torch.nn.DataParallel(model, device_ids = device_ids).to(device)
    criterion = torch.nn.DataParallel(criterion)
    if args.use_adversary:
        discriminator = torch.nn.DataParallel(discriminator, device_ids = device_ids).to(device)
        criterion_adv = torch.nn.DataParallel(criterion_adv)
        
print('Featurizing examples...', flush = True)
features_train = convert_examples_to_features(examples_train,
                                        Constants.MAX_SEQ_LEN, tokenizer, output_mode = 'classification')

features_eval = convert_examples_to_features(examples_eval,
                                        Constants.MAX_SEQ_LEN, tokenizer, output_mode = 'classification')

features_test = convert_examples_to_features(examples_test,
                                        Constants.MAX_SEQ_LEN, tokenizer, output_mode = 'classification')
training_set = MIMICDataset(features_train, 'train' , 'binary')
training_generator = data.DataLoader(training_set, shuffle = True,  batch_size = args.train_batch_size)

val_set = MIMICDataset(features_eval, 'val', 'binary')
val_generator = data.DataLoader(val_set, shuffle = False,  batch_size = args.eval_batch_size)

test_set = MIMICDataset(features_test, 'test', 'binary')
test_generator = data.DataLoader(test_set, shuffle = False,  batch_size = args.eval_batch_size)

print("Training with %s patients and %s sequences" %(train_df.shape[0], len(features_train)), flush = True)

num_train_epochs = args.max_num_epochs
learning_rate = args.lr
num_train_optimization_steps = len(training_generator) * num_train_epochs
warmup_proportion = 0.1

PREDICTOR_CHECKPOINT_PATH = os.path.join(args.output_dir, 'predictor.chkpt')
MODEL_CHECKPOINT_PATH = os.path.join(args.output_dir, 'model.chkpt')

actual_val = val_df[targets]

def merge_probs(probs, c):
    return (np.max(probs, axis = 0) + np.mean(probs, axis = 0)*len(probs)/float(c))/(1+len(probs)/float(c))

def merge_preds(prediction_dict, c=2):
    merged_preds = {}
    for i in prediction_dict:        
        merged_preds[i] = merge_probs(prediction_dict[i], c)
    return merged_preds

def evaluate_on_set(model, generator, predictor, c_val=2):
    '''
    Input: a pytorch data loader, whether the generator is an embedding or text generator
    Outputs:
        prediction_dict: a dictionary mapping note_id (str) to list of predicted probabilities
        merged_preds: a dictionary mapping note_id (str) to a single merged probability
        embs: a dictionary mapping note_id (str) to a numpy 2d array (shape num_seq * 768)
    '''

    model.eval()
    predictor.eval()
    if generator.dataset.gen_type == 'val':
        prediction_dict = {str(idx): [0]*row['num_seqs'] for idx, row in val_df.iterrows()}
        embs = {str(idx):np.zeros(shape = (row['num_seqs'], EMB_SIZE)) for idx, row in val_df.iterrows()}
    elif generator.dataset.gen_type == 'test':
        prediction_dict = {str(idx): [0]*row['num_seqs'] for idx, row in test_df.iterrows()}
        embs = {str(idx):np.zeros(shape = (row['num_seqs'], EMB_SIZE)) for idx, row in test_df.iterrows()}
    elif generator.dataset.gen_type == 'train':
        prediction_dict = {str(idx): [0]*row['num_seqs'] for idx, row in train_df.iterrows()}
        embs = {str(idx):np.zeros(shape = (row['num_seqs'], EMB_SIZE)) for idx, row in train_df.iterrows()}

    with torch.no_grad():
        for input_ids, input_mask, segment_ids, y, group, guid, other_vars in generator:
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            y = y.to(device)
            group = group.to(device)
            hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            bert_out = extract_embeddings(hidden_states, args.emb_method)

            for i in other_vars:
                bert_out = torch.cat([bert_out, i.float().unsqueeze(dim = 1).to(device)], 1)

            preds = predictor(bert_out).detach().cpu()

            for c,i in enumerate(guid):
                note_id, seq_id = i.split('-')
                prediction_dict[note_id][int(seq_id)] = preds[c, :].detach().cpu().numpy()
                embs[note_id][int(seq_id), :] = bert_out[c,:EMB_SIZE].detach().cpu().numpy()


    merged_preds = merge_preds(prediction_dict, c_val)

    return (prediction_dict, merged_preds, embs)

predictor = Classifier(**predictor_params).to(device)
if n_gpu > 1:
    predictor = torch.nn.DataParallel(predictor).cuda()

if not(args.use_adversary):
    param_optimizer = list(model.named_parameters()) + list(predictor.named_parameters())
else:
    param_optimizer = list(model.named_parameters()) + list(predictor.named_parameters()) + list(discriminator.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, #no weight decay
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

es = EarlyStopping(patience = args.es_patience)

optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=learning_rate,
                                warmup=warmup_proportion,
                                t_total=num_train_optimization_steps)
warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)
for epoch in range(1, num_train_epochs+1):
    # training
    model.train()
    predictor.train()
    if args.use_adversary:
        discriminator.train()
    running_loss = 0.0
    num_steps = 0
    with tqdm(total=len(training_generator), desc="Epoch %s"%epoch) as pbar:
        for input_ids, input_mask, segment_ids, y, group, _, other_vars in training_generator:
            if len(y) == 1: #prevents batchnorm error with one sample
                break
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            y = y.to(device)
            group = group.to(device)

            hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            bert_out = extract_embeddings(hidden_states, args.emb_method)

            for i in other_vars:
                bert_out = torch.cat([bert_out, i.float().unsqueeze(dim = 1).to(device)], 1)

            preds = predictor(bert_out)
            loss = criterion(preds, y)

            if args.use_adversary:
                adv_input = bert_out[:, :-len(other_vars)]
                adv_pred = discriminator(adv_input)
                adv_loss = criterion_adv(adv_pred, group)

            if n_gpu > 1:
                loss = loss.mean()
                if args.use_adversary:
                    adv_loss = adv_loss.mean()

            if args.use_adversary:
                loss += adv_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            num_steps += 1
            running_loss += loss.item()
            mean_loss = running_loss/num_steps

            pbar.update(1)
            pbar.set_postfix_str("Running Training Loss: %.5f" % mean_loss)

    # evaluate here
    model.eval()
    predictor.eval()
    val_loss = 0
    with torch.no_grad():
        checkpoints = {PREDICTOR_CHECKPOINT_PATH: predictor,
                    MODEL_CHECKPOINT_PATH: model}
        for input_ids, input_mask, segment_ids, y, group, guid, other_vars in val_generator:
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            y = y.to(device)
            group = group.to(device)
            hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            bert_out = extract_embeddings(hidden_states, args.emb_method)

            for i in other_vars:
                bert_out = torch.cat([bert_out, i.float().unsqueeze(dim = 1).to(device)], 1)

            preds = predictor(bert_out)
            loss = criterion(preds, y)
            if n_gpu > 1:
                loss = loss.mean()
                if args.use_adversary:
                    adv_loss = adv_loss.mean()

            if args.use_adversary:
                loss += adv_loss

            val_loss += loss.item()
        val_loss /= len(val_generator)

    print('Val loss: %s'%val_loss, flush = True)
    es(val_loss, checkpoints)
    if es.early_stop:
        break

print('Trained for %s epochs' % epoch)
predictor.load_state_dict(load_checkpoint(PREDICTOR_CHECKPOINT_PATH))
os.remove(PREDICTOR_CHECKPOINT_PATH)

model.load_state_dict(load_checkpoint(MODEL_CHECKPOINT_PATH))
os.remove(MODEL_CHECKPOINT_PATH)

auprcs = [] #one value for each in c grid
prediction_dict, _, _ = evaluate_on_set(model, val_generator, predictor)
for c_val in c_grid:
    merged_preds_val = merge_preds(prediction_dict, c_val)
    merged_preds_val_list = [merged_preds_val[str(i)] for i in actual_val.index]
    auprcs.append(average_precision_score(actual_val.values.astype(int), merged_preds_val_list))
print(auprcs, flush = True)
idx_max = np.argmax(auprcs)
opt_c = c_grid[idx_max]
print('val AUPRC:%.5f  optimal c: %s' %(auprcs[idx_max], opt_c))

prediction_dict_val, merged_preds_val, embs_val = evaluate_on_set(model, val_generator, predictor, c_val = opt_c)
merged_preds_val_list = [merged_preds_val[str(i)] for i in actual_val.index]

auprc = average_precision_score(actual_val.values.astype(int), merged_preds_val_list)
ll = log_loss(actual_val.values.astype(int), merged_preds_val_list)
roc = roc_auc_score(actual_val.values.astype(int), merged_preds_val_list)
print('AUPRC: %.5f' % auprc)
print('Log Loss: %.5f' % ll)
print('AUROC: %.5f' % roc)

prediction_dict_test, merged_preds_test, embs_test = evaluate_on_set(model, test_generator, predictor, c_val = opt_c)
# save predictor
json.dump(predictor_params, open(os.path.join(args.output_dir, 'predictor_params.json'), 'w'))
torch.save(predictor.state_dict(), os.path.join(args.output_dir, 'predictor.pt'))

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(args.output_dir)

json.dump(vars(args), open(os.path.join(args.output_dir, 'argparse_args.json'), 'w'))

rough_preds = {**merged_preds_val, **merged_preds_test}
pickle.dump(rough_preds, open(os.path.join(args.output_dir, 'preds.pkl'), 'wb'))
