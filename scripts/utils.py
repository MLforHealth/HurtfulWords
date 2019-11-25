import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils import data
import hashlib

def plot_training_history(hist_dict, metric, out_dir, title="", val_hist_dict=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # Plot training and validation accuracy
    ax.plot(range(1, len(hist_dict[metric]) + 1), hist_dict[metric])
    if val_hist_dict:
        ax.plot(range(1, len(val_hist_dict[metric]) + 1), val_hist_dict[metric])

    # Set plot titles, labels, ticks, and legend
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('Step')
    ax.set_xticks(np.arange(1, len(hist_dict[metric])+1), len(hist_dict[metric])/10)
    if val_hist_dict:
        ax.legend(['train', 'val'], loc='best')

    # Save plot
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(f"{out_dir}/{metric}.png")


def __nested_sorted_repr(c):
    if type(c) in (set, frozenset):
        return tuple(sorted(c))
    if type(c) is dict:
        return tuple(sorted([(k, __nested_sorted_repr(v)) for k, v in c.items()]))
    if type(c) in (tuple, list):
        return tuple([__nested_sorted_repr(e) for e in c])
    else:
        return c

def create_hdf_key(d):
    return hashlib.md5(str(__nested_sorted_repr(d)).encode()).hexdigest()

class Classifier(nn.Module):
    def __init__(self, input_dim, num_layers, dropout_prob, task_type, multiclass_nclasses = 0, decay_rate = 2):
        super(Classifier, self).__init__()
        self.task_type = task_type
        self.layers = []
        self.d = decay_rate
        for c, i in enumerate(range(num_layers)):
            if c != num_layers-1:
                self.layers.append(nn.Linear(input_dim // (self.d**c), input_dim // (self.d**(c+1))))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(input_dim // (self.d**(c+1))))
                self.layers.append(nn.Dropout(p = dropout_prob))
            else:
                if task_type == 'binary':
                    self.layers.append(nn.Linear(input_dim // (self.d**c), 1))
                    self.layers.append(nn.Sigmoid())
                elif task_type == 'multiclass':
                    self.layers.append(nn.Linear(input_dim // (self.d**c), multiclass_nclasses))
                    self.layers.append(nn.Softmax(dim = 1))
                elif task_type == 'regression':
                    self.layers.append(nn.Linear(input_dim // (self.d**c), 1))
                    self.layers.append(nn.ReLU())
                else:
                    raise Exception('Invalid task type!')

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        '''
        x: batch_size*input_dim
        output: batch_size*1
        '''
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x.squeeze(dim = 1)


def get_emb_size(emb_method):
    if emb_method == 'last' or emb_method == 'sum4':
        return 768
    elif emb_method == 'cat4':
        return 768 * 4
    else:
        raise Exception('Embedding method not supported!')

class MIMICDataset(data.Dataset):
    def __init__(self, features, gen_type, task_type):
        self.features = features
        self.gen_type = gen_type
        self.length = len(features)
        self.task_type = task_type

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        all_input_ids = torch.tensor(self.features[index].input_ids, dtype = torch.long)
        all_input_mask = torch.tensor(self.features[index].input_mask, dtype = torch.long)
        all_segment_ids = torch.tensor(self.features[index].segment_ids, dtype = torch.long)
        if self.task_type in ['binary', 'regression']:
            y = torch.tensor(self.features[index].label_id, dtype = torch.float32)
        else:
            y = torch.tensor(self.features[index].label_id, dtype = torch.long)
        group = torch.tensor(self.features[index].group, dtype = torch.long)
        guid = self.features[index].guid
        other_vars = self.features[index].other_fields

        return all_input_ids, all_input_mask, all_segment_ids, y, group, guid, other_vars


def extract_embeddings(v, emb_method):
    '''
    Given a BERT list of hidden layer states, extract the appropriate embedding
    '''
    if emb_method == 'last':
        return v[-1][:, 0, :] #last layer CLS token
    elif emb_method == 'sum4':
        return v[-1][:, 0, :] + v[-2][:, 0, :] + v[-3][:, 0, :] + v[-4][:, 0, :]
    elif emb_method == 'cat4':
        return torch.cat((v[-1][:, 0, :] , v[-2][:, 0, :] , v[-3][:, 0, :] , v[-4][:, 0, :]), 1)



#from Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, models): #models is a dict {path: model}

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            save_checkpoint(models)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_checkpoint(models)
            self.counter = 0

def save_checkpoint(models):
	for path in models:
		torch.save(models[path].state_dict(), path)

def load_checkpoint(path):
	return torch.load(path)

