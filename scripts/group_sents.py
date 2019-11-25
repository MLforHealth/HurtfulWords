#!/h/haoran/anaconda3/bin/python
import pandas as pd
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random
import argparse

parser = argparse.ArgumentParser('''Sentences from the sentence tokenizer can be very short. This script packs together several sentences into sequences
                                 to ensure that tokenA and tokenB have some minimum length (guaranteed except for sentences at the end of a document) when training BERT''')
parser.add_argument("input_loc", help = "pickled dataframe with 'sents' column", type=str)
parser.add_argument('output_loc', help = "path to output the dataframe", type=str)
parser.add_argument("model_path", help = 'folder with trained SciBERT model and tokenizer', type=str)
parser.add_argument("--under_prob", help = 'probability of being under the limit in a sequence', type=float, default = 0)
parser.add_argument('-m','--minlen', help = 'minimum lengths of tokens to pack the sentences into. Note that this is the length of a SINGLE sequence, not both', nargs = '+',
                     type=int,  dest='minlen', default = [20])
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case = True)

df = pd.read_pickle(args.input_loc)

def pack_sentences(row, minlen):
    i, cumsum, init = 0,0,0
    seqs, tok_len_sums = [], []
    while i<len(row.sent_toks_lens):
        cumsum += row.sent_toks_lens[i]
        if cumsum>= minlen:
            if init == i or random.random() >= args.under_prob:
                seqs.append('\n'.join(row.sents[init:i+1]))
            else: #roll back one
                seqs.append('\n'.join(row.sents[init:i]))
                cumsum -= row.sent_toks_lens[i]
                i -=1
            tok_len_sums.append(cumsum)
            cumsum = 0
            init = i+1
        i+=1
    if init != i:
        seqs.append('\n'.join(row.sents[init:]))
        tok_len_sums.append(cumsum)
    return [seqs, tok_len_sums]

for i in args.minlen:
	df['BERT_sents'+str(i)], df['BERT_sents_lens'+str(i)] = zip(*df.apply(pack_sentences, axis = 1, minlen = i))
	df['num_BERT_sents'+str(i)] = df['BERT_sents'+str(i)].apply(len)
	assert(all(df['BERT_sents_lens'+str(i)].apply(sum) == df['sent_toks_lens'].apply(sum)))

df.to_pickle(args.output_loc)
