#!/h/haoran/anaconda3/bin/python
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
from tqdm import tqdm
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
parser.add_argument("--model_path", type=str)
parser.add_argument('--fold_id', help = 'what fold to use as the DEV fold. Dataframe must have a "fold" column',nargs = '+', type=str, dest = 'fold_id', default = [])
parser.add_argument('--target_col_name', help = 'name of target to train on. Must be a column in the dataframe', type=str)
parser.add_argument("--output_dir", help = 'folder to output model/results', type=str)
parser.add_argument('--use_adversary', help = "whether or not to use an adversary. If True, must not have --freeze_bert", action = 'store_true')
parser.add_argument('--lm', help = 'lambda value for the adversary', type = float, default = 1.0)
parser.add_argument('--protected_group', help = 'name of protected group, must be a column in the dataframe', type = str, default = 'insurance')
parser.add_argument('--adv_layers', help = 'number of layers in adversary', type = int, default = 2)
parser.add_argument('--freeze_bert', help = 'freeze all BERT layers and only use pre-trained representation', action = 'store_true')
parser.add_argument('--train_batch_size', help = 'batch size to use for training', type = int)
parser.add_argument('--max_num_epochs', help = 'maximum number of epochs to train for', type = int, default = 20)
parser.add_argument('--es_patience', help = 'patience for the early stopping', type = int, default = 3)
parser.add_argument('--other_fields', help = 'other fields to add, must be columns in df', nargs = '+', type = str, dest = 'other_fields', default = [])
parser.add_argument('--seed', type = int, default = 42, help = 'random seed for initialization')
parser.add_argument('--dropout', type = float, default = 0, help = 'dropout probability for classifier')
parser.add_argument('--lr', type = float, default = 5e-4, help = 'learning rate for BertAdam optimizer')
parser.add_argument('--predictor_layers', type = int, default = 2, help = 'number of layers for classifier, ignored if gridsearch_classifier')
parser.add_argument('--emb_method', default = 'last', const = 'last', nargs = '?', choices = ['last', 'sum4', 'cat4'], help = 'what embedding layer to take')
parser.add_argument('--fairness_def', default = 'demo', const = 'demo', nargs = '?', choices = ['demo', 'odds'], help = 'what fairness definition to use: demographic parity, equality of odds')
parser.add_argument('--task_type', default = 'binary', const = 'binary', nargs = '?', choices = ['binary', 'multiclass', 'regression'], help = 'what type of data the target_col_name is')
parser.add_argument('--save_embs', help = 'save computed embeddings at the end', action = 'store_true')
parser.add_argument('--output_train_stats', help = 'export training set predictions into the dataframe', action = 'store_true')
parser.add_argument('--gridsearch_classifier', help = 'whether to run a grid search over the classifier parameters, using AUPRC as metric', action = 'store_true')
parser.add_argument('--average', help = 'whether to aggregate sequences to a single prediction by simple average, or by using the NYU agg function', action = 'store_true')
parser.add_argument('--gridsearch_c', help = 'whether to run a grid search over the NYU agg c parameter, using AUPRC as metric, only valid if not --average, and --gridsearch_classifier', action = 'store_true')
parser.add_argument('--use_new_mapping', help = 'whether to use new mapping for adversarial training', action = 'store_true')
parser.add_argument('--pregen_emb_path', help = '''if embeddings have been precomputed, can provide a path here (as a pickled dictionary mapping note_id:numpy array).
                        Will only be used if freeze_bert. note_ids in this dictionary must a be a superset of the note_ids in df_path''', type = str)
parser.add_argument('--overwrite', help = 'whether to overwrite existing model/predictions', action = 'store_true')
args = parser.parse_args()

if os.path.isfile(os.path.join(args.output_dir, 'preds.pkl')) and not args.overwrite:
    print("File already exists; exiting.")
    sys.exit()

print('Reading dataframe...', flush = True)
df = pd.read_pickle(args.df_path)
if 'note_id' in df.columns:
    df = df.set_index('note_id')

tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertModel.from_pretrained(args.model_path)

target = args.target_col_name
assert(target in df.columns)

#even if no adversary, must have valid protected group column for code to work
if args.use_adversary:
    protected_group = args.protected_group
    assert(protected_group in df.columns)
    if args.use_new_mapping:
        mapping = Constants.newmapping
        for i in Constants.drop_groups[protected_group]:
            df = df[df[protected_group] != i]
    else:
        mapping = Constants.mapping

other_fields_to_include = args.other_fields
if args.freeze_bert:
    for param in model.parameters():
        param.requires_grad = False

assert('fold' in df.columns)
for i in args.fold_id:
	assert(i in df['fold'].unique())
assert('test' in df['fold'].unique())
fold_id = args.fold_id

if args.gridsearch_c:
    assert(args.task_type == 'binary')
    c_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000]
else:
    c_grid = [2]

Path(args.output_dir).mkdir(parents = True, exist_ok = True)

EMB_SIZE = get_emb_size(args.emb_method)
train_df = df[~df.fold.isin(['test', 'NA', *fold_id])]
val_df = df[df.fold.isin(fold_id)]
test_df = df[df.fold == 'test']

def convert_input_example(note_id, text, seqIdx, target, group, other_fields = []):
    return InputExample(guid = '%s-%s'%(note_id,seqIdx), text_a = text, text_b = None, label = target, group = mapping[protected_group][group] if args.use_adversary else 0, other_fields = other_fields)

# in training generator, return all folds except this.
# in validation generator, return only this fold

print('Converting input examples to appropriate format...', flush = True)
examples_train = [convert_input_example(idx, i, c, row[target], row[protected_group] if args.use_adversary else 0,
                                       [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in train_df.iterrows()
                  for c, i in enumerate(row.seqs)]

examples_eval = [convert_input_example(idx, i, c, row[target], row[protected_group] if args.use_adversary else 0,
                                      [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in val_df.iterrows()
                  for c, i in enumerate(row.seqs)]

examples_test = [convert_input_example(idx, i, c, row[target], row[protected_group] if args.use_adversary else 0,
                                      [] if len(other_fields_to_include) ==0 else row[other_fields_to_include].values.tolist())
                  for idx, row in test_df.iterrows()
                  for c, i in enumerate(row.seqs)]

def convert_examples_to_features_emb(examples, embs):
    features = []
    for i in examples:
        note_id, seq_id = i.guid.split('-')
        emb = embs[note_id][int(seq_id), :]
        features.append(EmbFeature(emb, y = i.label, guid = i.guid, group = i.group, other_fields = i.other_fields))
    return features

class EmbFeature():
    def __init__(self, emb, y, guid, group, other_fields):
        self.emb = emb
        self.y = y
        self.guid = guid
        self.group = group
        self.other_fields = other_fields

class Embdataset(data.Dataset):
    def __init__(self, features, gen_type):
        self.features = features #list of EmbFeatures
        self.gen_type = gen_type
        self.length = len(features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        emb = torch.tensor(self.features[index].emb, dtype = torch.float32)
        if args.task_type in ['binary', 'regression']:
            y = torch.tensor(self.features[index].y, dtype = torch.float32)
        else:
            y = torch.tensor(self.features[index].y, dtype = torch.long)
        other_fields = self.features[index].other_fields
        guid = self.features[index].guid

        return emb, y, guid, other_fields


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

if args.gridsearch_classifier:
    assert(args.freeze_bert)
    grid = list(ParameterGrid({
        'num_layers': [2,3,4],
        'dropout_prob': [0, 0.2],
        'decay_rate': [2,4,6]
    }))
    grid.append({
        'num_layers': 1,
        'dropout_prob': 0,
        'decay_rate': 2
        })
    for i in grid: # adds extra fields to input arguments
        i['input_dim'] = EMB_SIZE + len(other_fields_to_include)
        i['task_type'] = args.task_type
else:
    grid = [{ # only one parameter combination
        'input_dim': EMB_SIZE + len(other_fields_to_include),
        'num_layers': args.predictor_layers,
        'dropout_prob': args.dropout,
        'task_type': args.task_type
    }]

if args.task_type == 'multiclass':
    for i in grid:
        i['multiclass_nclasses'] = len(df[target].unique())

if args.use_adversary:
	discriminator = Discriminator(EMB_SIZE + int(args.fairness_def == 'odds'), args.adv_layers, len(mapping[protected_group]), args.lm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
model.to(device)
if args.use_adversary:
    discriminator.to(device)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if args.task_type == 'binary':
    criterion = nn.BCELoss()
elif args.task_type == 'multiclass':
    criterion = nn.CrossEntropyLoss()
elif args.task_type == 'regression':
    criterion = nn.MSELoss()

criterion_adv = nn.CrossEntropyLoss()

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    criterion = torch.nn.DataParallel(criterion)
    if args.use_adversary:
        discriminator = torch.nn.DataParallel(discriminator)
        criterion_adv = torch.nn.DataParallel(criterion_adv)


def get_embs(generator):
    '''
    given a generator, runs all the data through one pass of the model to calculate embeddings
    used when BERT weights are frozen, calculates embeddings first to save compute
    '''
    features = []
    model.eval()
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, y, group, guid, other_vars in generator:
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            bert_out = extract_embeddings(hidden_states, args.emb_method)

            for c,i in enumerate(guid):
                note_id, seq_id = i.split('-')
                emb = bert_out[c,:].detach().cpu().numpy()
                features.append(EmbFeature(emb = emb, y = y[c], guid = i, group = group, other_fields= [i[c] for i in other_vars]))
    return features


print('Featurizing examples...', flush = True)
if not args.pregen_emb_path:
    features_train = convert_examples_to_features(examples_train,
                                            Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))

    features_eval = convert_examples_to_features(examples_eval,
                                            Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))

    features_test = convert_examples_to_features(examples_test,
                                            Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))

    training_set = MIMICDataset(features_train, 'train' ,args.task_type)
    training_generator = data.DataLoader(training_set, shuffle = True,  batch_size = args.train_batch_size, drop_last = True)

    val_set = MIMICDataset(features_eval, 'val', args.task_type)
    val_generator = data.DataLoader(val_set, shuffle = False,  batch_size = args.train_batch_size)

    test_set = MIMICDataset(features_test, 'test', args.task_type)
    test_generator = data.DataLoader(test_set, shuffle = False,  batch_size = args.train_batch_size)

if args.freeze_bert: #only need to precalculate for training and val set
    if args.pregen_emb_path:
        pregen_embs = pickle.load(open(args.pregen_emb_path, 'rb'))
        features_train_embs = convert_examples_to_features_emb(examples_train, pregen_embs)
        features_val_embs = convert_examples_to_features_emb(examples_eval, pregen_embs)
        features_test_embs = convert_examples_to_features_emb(examples_test, pregen_embs)
    else:
        features_train_embs = get_embs(training_generator)
        features_val_embs = get_embs(val_generator)
        features_test_embs = get_embs(test_generator)
    training_generator = data.DataLoader(Embdataset(features_train_embs, 'train'), shuffle = True, batch_size = args.train_batch_size, drop_last = True)
    val_generator = data.DataLoader(Embdataset(features_val_embs, 'val'), shuffle = False,  batch_size = args.train_batch_size)
    test_generator= data.DataLoader(Embdataset(features_test_embs, 'test'), shuffle = False,  batch_size = args.train_batch_size)

num_train_epochs = args.max_num_epochs
learning_rate = args.lr
num_train_optimization_steps = len(training_generator) * num_train_epochs
warmup_proportion = 0.1

PREDICTOR_CHECKPOINT_PATH = os.path.join(args.output_dir, 'predictor.chkpt')
MODEL_CHECKPOINT_PATH = os.path.join(args.output_dir, 'model.chkpt')

grid_auprcs = []
es_models = []
optimal_cs = []
actual_val = val_df[target]

def merge_probs(probs, c):
    return (np.max(probs) + np.mean(probs)*len(probs)/float(c))/(1+len(probs)/float(c))

def avg_probs(probs):
    return np.mean(probs)

def avg_probs_multiclass(probs):
    return np.argmax(np.mean(probs, axis = 0))

def merge_regression(preds):
    return np.mean(preds)

def evaluate_on_set(generator, predictor, emb_gen = False, c_val=2):
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

    if emb_gen:
        with torch.no_grad():
            for embs, y, guid, other_vars in generator:
                embs = embs.to(device)
                y = y.to(device)
                for i in other_vars:
                    embs = torch.cat([embs, i.float().unsqueeze(dim = 1).to(device)], 1)
                preds = predictor(embs).detach().cpu()
                for c,i in enumerate(guid):
                    note_id, seq_id = i.split('-')
                    if args.task_type in ['binary', 'regression']:
                        prediction_dict[note_id][int(seq_id)] = preds[c].item()
                    else:
                        prediction_dict[note_id][int(seq_id)] = preds[c,:].numpy()

    else:
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
                    if args.task_type in ['binary', 'regression']:
                        prediction_dict[note_id][int(seq_id)] = preds[c].item()
                    else:
                        prediction_dict[note_id][int(seq_id)] = preds[c,:].numpy()
                    embs[note_id][int(seq_id), :] = bert_out[c,:EMB_SIZE].detach().cpu().numpy()


    merged_preds = merge_preds(prediction_dict, c_val)

    return (prediction_dict, merged_preds, embs)

def merge_preds(prediction_dict, c=2):
    merged_preds = {}
    for i in prediction_dict:
        if args.task_type ==  'binary':
            if args.average:
                merged_preds[i] = avg_probs(prediction_dict[i])
            else:
                merged_preds[i] = merge_probs(prediction_dict[i], c)
        elif args.task_type == 'regression':
            merged_preds[i] = merge_regression(prediction_dict[i])
        elif args.task_type == 'multiclass':
            merged_preds[i] = avg_probs_multiclass(np.array(prediction_dict[i]))
    return merged_preds

for predictor_params in grid:
    print(predictor_params, flush = True)
    predictor = Classifier(**predictor_params).to(device)
    if n_gpu > 1:
        predictor = torch.nn.DataParallel(predictor)

    if not(args.freeze_bert) and not(args.use_adversary):
        param_optimizer = list(model.named_parameters()) + list(predictor.named_parameters())
    elif args.freeze_bert and not(args.use_adversary):
        param_optimizer = list(predictor.named_parameters())
    elif args.freeze_bert and args.use_adversary:
        raise Exception('No purpose in using an adversary if BERT layers are frozen')
    else:
        param_optimizer = list(model.named_parameters()) + list(predictor.named_parameters()) + list(discriminator.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
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
        if not args.freeze_bert:
            model.train()
        else:
            model.eval()
        predictor.train()
        if args.use_adversary:
            discriminator.train()
        running_loss = 0.0
        num_steps = 0
        with tqdm(total=len(training_generator), desc="Epoch %s"%epoch) as pbar:
            if not args.freeze_bert:
                for input_ids, input_mask, segment_ids, y, group, _, other_vars in training_generator:
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
                        if args.fairness_def == 'odds':
                            adv_input = torch.cat([adv_input, y.unsqueeze(dim = 1)], 1)
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
            else: # if frozen, use precomputed embeddings to save time
                for embs, y,_, other_vars in training_generator:
                    embs = embs.to(device)
                    y = y.to(device)
                    for i in other_vars:
                        embs = torch.cat([embs, i.float().unsqueeze(dim = 1).to(device)], 1)
                    preds = predictor(embs)
                    loss = criterion(preds, y)

                    if n_gpu > 1:
                        loss = loss.mean()
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
            if args.freeze_bert:
                checkpoints = {PREDICTOR_CHECKPOINT_PATH: predictor}
                for embs, y, guid, other_vars in val_generator:
                    embs = embs.to(device)
                    y = y.to(device)
                    for i in other_vars:
                        embs = torch.cat([embs, i.float().unsqueeze(dim = 1).to(device)], 1)
                    preds = predictor(embs)
                    loss = criterion(preds, y)

                    if n_gpu > 1:
                        loss = loss.mean()
                    val_loss += loss.item()

                val_loss /= len(val_generator)
                # early stopping uses val loss as metric
                # model selection/c selection uses AUPRC as metric
            else:
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
    if not args.freeze_bert:
        model.load_state_dict(load_checkpoint(MODEL_CHECKPOINT_PATH))
        os.remove(MODEL_CHECKPOINT_PATH)

    if args.gridsearch_classifier:
        auprcs = [] #one value for each in c grid
        prediction_dict, _, _ = evaluate_on_set(val_generator, predictor, emb_gen = args.freeze_bert)
        for c_val in c_grid:
            merged_preds_val = merge_preds(prediction_dict, c_val)
            merged_preds_val_list = [merged_preds_val[str(i)] for i in actual_val.index]
            auprcs.append(average_precision_score(actual_val.values.astype(int), merged_preds_val_list))
        print(auprcs, flush = True)
        print(c_grid, flush = True)
        idx_max = np.argmax(auprcs)
        grid_auprcs.append(auprcs[idx_max])
        es_models.append(predictor.cpu())
        optimal_cs.append(c_grid[idx_max])
        print('val AUPRC:%.5f  optimal c: %s' %(auprcs[idx_max], c_grid[idx_max] ))

# find best predictor here, move back to cpu
if args.gridsearch_classifier:
    idx_max = np.argmax(grid_auprcs)
    predictor = es_models[idx_max].to(device)
    opt_c = optimal_cs[idx_max]
else:
    opt_c = 2.0

# evaluate on val set
prediction_dict_val, merged_preds_val, embs_val = evaluate_on_set(val_generator, predictor, emb_gen = args.freeze_bert, c_val = opt_c)
merged_preds_val_list = [merged_preds_val[str(i)] for i in actual_val.index]

if args.task_type == 'binary':
	acc = accuracy_score(actual_val.values.astype(int), np.array(merged_preds_val_list).round())
	auprc = average_precision_score(actual_val.values.astype(int), merged_preds_val_list)
	ll = log_loss(actual_val.values.astype(int), merged_preds_val_list)
	roc = roc_auc_score(actual_val.values.astype(int), merged_preds_val_list)
	print('Accuracy: %.5f' % acc)
	print('AUPRC: %.5f' % auprc)
	print('Log Loss: %.5f' % ll)
	print('AUROC: %.5f' % roc)
elif args.task_type == 'regression':
	mse = mean_squared_error(actual_val, merged_preds_val_list)
	print('MSE: %.5f' % mse)
elif args.task_type == 'multiclass':
	report = classification_report(actual_val.values.astype(int), np.array(merged_preds_val_list))
	print(report)

prediction_dict_test, merged_preds_test, embs_test = evaluate_on_set(test_generator, predictor, emb_gen = args.freeze_bert,  c_val = opt_c)
if args.output_train_stats:
    prediction_dict_train, merged_preds_train, embs_train = evaluate_on_set(training_generator, predictor, emb_gen = args.freeze_bert, c_val = opt_c)
else:
    merged_preds_train, embs_train = {}, {}

# save predictor
json.dump(predictor_params, open(os.path.join(args.output_dir, 'predictor_params.json'), 'w'))
torch.save(predictor.state_dict(), os.path.join(args.output_dir, 'predictor.pt'))

# save model
if not args.freeze_bert:
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

# save args
json.dump(vars(args), open(os.path.join(args.output_dir, 'argparse_args.json'), 'w'))

#saves embeddings
if args.save_embs:
    embs = {**embs_val, **embs_test, **embs_train}
    pickle.dump(embs, open(os.path.join(args.output_dir, 'embs.pkl'), 'wb'))

rough_preds = {**merged_preds_val, **merged_preds_test, **merged_preds_train}
pickle.dump(rough_preds, open(os.path.join(args.output_dir, 'preds.pkl'), 'wb'))

# saves gridsearch info
pickle.dump({
'grid_auprcs':grid_auprcs,
'optimal_cs': optimal_cs,
'opt_c': opt_c
    }, open(os.path.join(args.output_dir, 'gs_info.pkl'), 'wb'))

