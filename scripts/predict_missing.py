import torch
from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import numpy as np
import random

###########################
# CONFIGURATIONS
###########################

SCIBERT_DIR = Path('/scratch/hdd001/home/haoran/shared_data/scibert_scivocab_uncased/')

RACE_LIST = ['caucasian', 'hispanic', 'african', 'african american', 'white']
###########################

def get_words_for_blank_slow_decode(text: str, model: BertForMaskedLM, tokenizer: BertTokenizer):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


    mask_positions = []
    tokenized_text = tokenizer.tokenize(text)
    top_words_all = []
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '_':
            tokenized_text[i] = '[MASK]'
            mask_positions.append(i)

    while mask_positions:
        top_words = []
        # Convert tokens to vocab indices
        token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([token_ids])

        # Call BERT to calculate unnormalized probabilities for all pos
        model.eval()
        predictions = model(tokens_tensor)

        # get predictions
        mask_preds = predictions[0, mask_positions, :]

        candidates = [] #(word, prob)
        for mask_pos in mask_positions:
            mask_preds = predictions[0, mask_pos, :]

            top_idxs = mask_preds.detach().numpy().argsort()[::-1]
            top_idx = top_idxs[0]
            top_prob = mask_preds[top_idx]
            top_word = tokenizer.ids_to_tokens[top_idx]
            candidates.append((top_word, top_prob.detach().item()))
            top_words_pos = []
            for i in top_idxs[:20]:
                top_words_pos.append((tokenizer.ids_to_tokens[i], mask_preds[i].detach().item()))
            top_words.append(top_words_pos)
        best_candidate = max(candidates, key = lambda x: x[1])
        best_pos = mask_positions[candidates.index(best_candidate)]

        tokenized_text[best_pos] = best_candidate[0]
        mask_positions = [i for i in mask_positions if i != best_pos]

        top_words_all.append(top_words[candidates.index(best_candidate)])

    pred_sent = ' '.join(tokenized_text).replace(' ##', '')
    return (pred_sent, top_words_all)


if __name__ == "__main__":
    templates = []
    templates.append("[CLS] [RACE] pt became belligerent and violent. sent to _ _ [SEP]")

    # Replace the "demographic keyword" with those previous specified
    sentences = []
    for line in templates:
        if '[RACE]' in line:
            for race in RACE_LIST:
                new_text = line.replace('[RACE]', race)
                sentences.append(new_text)


    # Load pre-trained model with masked language model head
    model = BertForMaskedLM.from_pretrained(SCIBERT_DIR)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(SCIBERT_DIR)

    # fills in the missing word
    for sent in sentences:
        s, t = get_words_for_blank_slow_decode(sent, model, tokenizer)
        print(s)
