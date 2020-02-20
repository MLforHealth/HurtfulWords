'''Adapted from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/finetune_on_pregenerated.py'''
import sys
import os
sys.path.insert(0, os.getcwd())
from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import copy
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from gradient_reversal import GradientReversal
import Constants
import utils


# Create the InputFeatures container with named tuple fields
InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next domain_a domain_b")

# Configure loggers
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def convert_example_to_features(example, tokenizer, domain_to_id_dict, domain_name, max_seq_length):
    '''Helper function for turning JSON strings into tokenized features'''
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    groups_a = example["groups_a"]  # dictionary of 5 protected group categories and associated attribute
    groups_b = example["groups_b"]  # same keys as above, but with values for the second sequence
    domain_a = domain_to_id_dict[groups_a[domain_name]]  # for the domain of interest, convert the category string to unique category ID
    domain_b = domain_to_id_dict[groups_b[domain_name]]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next,
                             domain_a=domain_a,
                             domain_b=domain_b,
                            )
    return features

def _save_model(model, args, suffix, config_suffix="", save_config=False):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, suffix)
    torch.save(model_to_save.state_dict(), output_model_file)
    if save_config:
        output_config_file = os.path.join(args.output_dir, config_suffix)
        model_to_save.config.to_json_file(output_config_file)

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, domain_to_id_dict, domain_name, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.domain_to_id_dict = domain_to_id_dict
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
            domain_a = np.memmap(filename=self.working_dir/'domain_a.memmap',
                                shape=(num_samples,), mode='w+', dtype=np.int32)
            domain_b = np.memmap(filename=self.working_dir/'domain_b.memmap',
                                shape=(num_samples,), mode='w+', dtype=np.int32)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
            domain_a = np.zeros(shape=(num_samples,), dtype=np.int32)
            domain_b = np.zeros(shape=(num_samples,), dtype=np.int32)
        logging.info(f"Loading training examples for epoch {epoch}Þ[MaÞ[MaÞ")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, domain_to_id_dict, domain_name, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                domain_a[i] = features.domain_a
                domain_b[i] = features.domain_b
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        self.domain_a = domain_a
        self.domain_b = domain_b

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)),
                torch.tensor(self.domain_a[item].astype(np.int64)),
                torch.tensor(self.domain_b[item].astype(np.int64)))


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
                self.layers.append(nn.Softmax())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--domain_of_interest', type=str, required=True)  # Added for domain adaptation
    parser.add_argument('--layer_to_get_features', type=int, default=11, help="Choose an integer in [0, 11] for BERT basic (with 12 layers) or [0, 23] for BERT large (with 24 layers)")
    parser.add_argument('--discriminator_input_dim', type=int, default=768, help='Must correspond to number of hidden dimensions for BERT embeddings.')
    parser.add_argument('--lambda_', type=float, required=True, help = 'Weighting parameter for the loss of the adversarial network')
    parser.add_argument('--num_layers', type=int, required=True, help = 'Number of fully connected layers for the discriminator')
    parser.add_argument("--use_new_mapping", action="store_true", help = 'whether to use new mapping in Constants file')
    parser.add_argument('--discriminator_a_path', type=str, required = False, help = 'path for pretrained discriminator_a if it exists, otherwise initialize from random')
    parser.add_argument('--discriminator_b_path', type=str, required = False, help = 'path for pretrained discriminator_b if it exists, otherwise initialize from random')
    parser.add_argument("--bert_model", type=str, required=True, help="Path to BERT pre-trained model, or select from list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    # get domain mapping
    assert (args.domain_of_interest in Constants.mapping)
    if args.use_new_mapping:
        domain_mapping = Constants.newmapping[args.domain_of_interest]
    else:
        domain_mapping = Constants.mapping[args.domain_of_interest]
    num_categories = len(set(domain_mapping.values()))

    # check that data has been pregenerated for the specified epochs
    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    # get up GPU
    if args.local_rank == -1 or args.no_cuda:
        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device("cuda")
        else:
            print("[WARNING] Using CPU instead of GPU for training!")
            device = torch.device("cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Create transformer encoder layers
    model = BertForPreTraining.from_pretrained(args.bert_model)
    embed_layers = model.bert
    embed_layers = embed_layers.to(device)

    # Make discriminator networks for the two sentences
    discriminator_a = Discriminator(input_dim = args.discriminator_input_dim,
                                  num_layers = args.num_layers,
                                  num_categories = num_categories,
                                  lm = args.lambda_)
    discriminator_b = Discriminator(input_dim = args.discriminator_input_dim,
                                  num_layers = args.num_layers,
                                  num_categories = num_categories,
                                  lm = args.lambda_)


    # Prepare models for GPU training
    if args.fp16:
        # cast floating point parameters to the half precision datatype
        model.half()
        discriminator_a.half()
        discriminator_b.half()

    model = model.to(device)
    discriminator_a = discriminator_a.to(device)
    discriminator_b = discriminator_b.to(device)

    if args.discriminator_a_path:
        discriminator_a.load_state_dict(torch.load(args.discriminator_a_path))

    if args.discriminator_b_path:
        discriminator_b.load_state_dict(torch.load(args.discriminator_b_path))

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
        discriminator_a = DPP(model)
        discriminator_b = DPP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        discriminator_a = torch.nn.DataParallel(discriminator_a)
        discriminator_b = torch.nn.DataParallel(discriminator_b)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters()) + list(discriminator_a.named_parameters()) + list(discriminator_b.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    loss_func = nn.DataParallel(nn.CrossEntropyLoss())

    # Track training accuracy/loss across **all epochs**
    train_hist = {'domain_a_loss': [],
                'domain_a_acc': [],
                'domain_b_loss': [],
                'domain_b_acc': [],
                'label_loss': [],
                'tr_loss': [],
                }

    for epoch in range(args.epochs):
        # put models in train mode
        model.train()
        discriminator_a.train()
        discriminator_b.train()

        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            domain_to_id_dict = domain_mapping, domain_name = args.domain_of_interest,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        # Reinitialize and reaccumulate stats for **current epoch** (i.e. across batchs)
        epoch_stats = {'domain_a_loss': 0,  # not yet normalized for the number of steps
                'domain_a_correct': 0,
                'domain_b_loss': 0,
                'domain_b_correct': 0,
                'label_loss': 0,
                'tr_loss': 0,
                'nb_tr_examples': 0,
                'nb_tr_steps': 0,
                }

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, domain_a, domain_b = batch

                # Get class label loss
                label_loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)

                # Get feature embeddings for this batch
                with torch.no_grad():
                    encoded_layers, pooled_output = embed_layers(input_ids, segment_ids, output_all_encoded_layers=True)
                features = encoded_layers[args.layer_to_get_features]
                assert features.shape[2] == args.discriminator_input_dim

                # We only feed the [CLS] token into the discriminator:
                domain_input = features[:, 0, :]   # tensor of (batch_size, hidden_dim)

                # get domain predictions and loss
                domain_input = domain_input.to(device)
                domain_a_preds = discriminator_a(domain_input)
                domain_b_preds = discriminator_b(domain_input)

                domain_a_loss = loss_func(domain_a_preds, domain_a)
                domain_b_loss = loss_func(domain_b_preds, domain_b)

                if n_gpu > 1:
                    label_loss = label_loss.mean() # mean() to average on multi-gpu.
                    domain_a_loss = domain_a_loss.mean()
                    domain_b_loss = domain_b_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    label_loss = label_loss / args.gradient_accumulation_steps
                    domain_a_loss = domain_a_loss / args.gradient_accumulation_steps
                    domain_b_loss = domain_b_loss / args.gradient_accumulation_steps
                loss = label_loss + domain_a_loss + domain_b_loss


                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # Gather loss and domain prediction accuracies
                domain_a_correct = domain_a_preds.argmax(dim=-1).eq(domain_a).sum().item()
                domain_b_correct = domain_b_preds.argmax(dim=-1).eq(domain_b).sum().item()

                epoch_stats['domain_a_loss'] += domain_a_loss.item()
                epoch_stats['domain_a_correct'] += domain_a_correct
                epoch_stats['domain_b_loss'] += domain_b_loss.item()
                epoch_stats['domain_b_correct'] += domain_b_correct
                epoch_stats['label_loss'] += label_loss.item()
                epoch_stats['tr_loss'] += loss.item()
                epoch_stats['nb_tr_examples'] += input_ids.size(0)
                epoch_stats['nb_tr_steps'] += 1

                pbar.update(1)
                mean_loss = epoch_stats['tr_loss'] * args.gradient_accumulation_steps / epoch_stats['nb_tr_steps']
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # At the end of **each minibatch**, add a new point to the training history plot
                # Note that even though a new point is plotted, the loss and accuracy is calculated from
                # stats accumulated across the entire epoch.
                train_hist['domain_a_loss'].append(epoch_stats['domain_a_loss'] / epoch_stats['nb_tr_steps'])
                train_hist['domain_b_loss'].append(epoch_stats['domain_b_loss'] / epoch_stats['nb_tr_steps'])
                train_hist['domain_a_acc'].append(epoch_stats['domain_a_correct'] / epoch_stats['nb_tr_examples'])
                train_hist['domain_b_acc'].append(epoch_stats['domain_b_correct'] / epoch_stats['nb_tr_examples'])
                train_hist['label_loss'].append(epoch_stats['label_loss'] / epoch_stats['nb_tr_steps'])
                train_hist['tr_loss'].append(epoch_stats['tr_loss'] / epoch_stats['nb_tr_steps'])

        # At the end of each **epoch**, save a trained model and a training loss/acc plot
        # Note that the plot will still accumulate points from _all_ epochs
        logging.info(f"** ** * Saving fine-tuned model for epoch {epoch} ** ** * ")
        _save_model(model, args, suffix=WEIGHTS_NAME, config_suffix=CONFIG_NAME, save_config=True)
        _save_model(discriminator_a, args, suffix=f"discriminator_a_{epoch}.bin", save_config=False)
        _save_model(discriminator_b, args, suffix=f"discriminator_b_{epoch}.bin", save_config=False)
        tokenizer.save_vocabulary(args.output_dir)

        utils.plot_training_history(train_hist, 'domain_a_loss', args.output_dir / 'figures', title="domain of first sequence: training loss")
        utils.plot_training_history(train_hist, 'domain_b_loss', args.output_dir / 'figures', title="domain of second sequence: training loss")
        utils.plot_training_history(train_hist, 'label_loss', args.output_dir / 'figures', title="BERT pretraining task labels: training loss")
        utils.plot_training_history(train_hist, 'tr_loss', args.output_dir / 'figures', title="Overall loss")
        utils.plot_training_history(train_hist, 'domain_a_acc', args.output_dir / 'figures', title="domain of first sequence: training accuracy")
        utils.plot_training_history(train_hist, 'domain_b_acc', args.output_dir / 'figures', title="domain of second sequence: training accuracy")

    print(f'Finished {args.epochs} epochs of training.')

    args_dict = copy.deepcopy(vars(args))
    for key, value in args_dict.items():
        args_dict[key] = str(args_dict[key])
    args_json = json.dumps(args_dict)
    with open(args.output_dir / "parser_arguments.json", "w") as f:
        for line in args_json:
            f.write(line)



if __name__ == '__main__':
    main()
