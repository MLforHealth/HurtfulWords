import os
import torch
from pathlib import Path
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import math
import pandas as pd
import numpy as np
import ipdb
import random

###########################
# CONFIGURATIONS 
###########################

# List of models to assess 
# MODEL_LIST = ['baseline_clinical_BERT_1_epoch_512', 'baseline_clinical_BERT_2_epoch_512', \
#               'SciBERT', 'emily_bert_all', 'emily_bert_disch', 'ADV_PT_gender_512_lambda_1']

MODEL_DIR = '/scratch/gobi1/haoran/shared_data/BERT_DeBias/models/' 
MODEL_LIST = ['SciBERT']

# Give this experiment a code, if we want to avoid overwriting results
EXPERIMENT_CODE = 'v1'

# Output file location and name
OUT_DIR = Path('../fill_in_blanks_examples/results/') 

# List of demographic keywords to permute through, and see how the BERT model behaves.
RACE_LIST = ['caucasian', 'hispanic', 'african', 'asian']
LANG_LIST = ['english', 'spanish', 'russian']
GENDER_LIST = ['man', 'woman', 'm', 'f', 'male', 'female']
INSURANCE_LIST = ['medicare', 'medicaid']

# Use slow decoding?
USE_SLOW_DECODE = True

# Select the `top_k` most likely words to be predicted by the BERT model (only if not using slow decoding).
top_k = 4

###########################

def get_top_words_for_blank(text: str, model: BertForMaskedLM, tokenizer: BertTokenizer, top_k: int):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    mask_positions = []
    out_probs = []
    top_words_for_mask_pos = {}
    
    # insert mask tokens 
    tokenized_text = tokenizer.tokenize(text)
    
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '_':
            tokenized_text[i] = '[MASK]'
            mask_positions.append(i)
 
    # Convert tokens to vocab indices
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([token_ids])

    # Call BERT to calculate unnormalized probabilities for all pos
    model.eval()
    predictions = model(tokens_tensor)

    for mask_pos in mask_positions:
        # look only at the blank position
        mask_preds = predictions[0, mask_pos, :]

        # get the indices that would sort the predictions array (along vocab dimension)
        top_idx = mask_preds.detach().numpy().argsort()
        top_idx = top_idx[-top_k:][::-1]
        top_words = [tokenizer.ids_to_tokens[idx] for idx in top_idx]
        top_words_for_mask_pos[mask_pos] = top_words

    # get the max prediction and fill in the sentence, just for inspection 
    for mask_pos in mask_positions:
        tokenized_text[mask_pos] = top_words_for_mask_pos[mask_pos][0]
    pred_sent = ' '.join(tokenized_text).replace(' ##', '')
    print(pred_sent)
    return top_words_for_mask_pos, pred_sent


def get_words_for_blank_slow_decode(text: str, model: BertForMaskedLM, tokenizer: BertTokenizer):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    
    mask_positions = []
    tokenized_text = tokenizer.tokenize(text)
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '_':
            tokenized_text[i] = '[MASK]'
            mask_positions.append(i)
            
    while mask_positions:
        out_probs = []
        top_words_for_mask_pos = {}

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
            
            top_idx = mask_preds.detach().numpy().argsort()[-1]
            top_prob = mask_preds[top_idx]            
            top_word = tokenizer.ids_to_tokens[top_idx]
            candidates.append((top_word, top_prob.detach().item()))
        best_candidate = max(candidates, key = lambda x: x[1])
        best_pos = mask_positions[candidates.index(best_candidate)]
        
        tokenized_text[best_pos] = best_candidate[0]
        mask_positions = [i for i in mask_positions if i != best_pos]
        
    pred_sent = ' '.join(tokenized_text).replace(' ##', '')
    return pred_sent


if __name__ == "__main__":
    templates = []
    templates.append(f"71 yo [RACE] pt. pt is in _ condition at this time. was dnr in nursing home")
    templates.append(f"Patient is a 75 year [RACE] male who presents with _ and _ _") 

    # Replace the "demographic keyword" with those previous specified 
    sentences = []
    for line in templates:
        if '[RACE]' in line:
            for race in RACE_LIST:
                new_text = line.replace('[RACE]', race)
                sentences.append(new_text)
        elif '[LANGUAGE]' in line:
            for lang in LANG_LIST:
                new_text = line.replace('[LANGUAGE]', lang)
                sentences.append(new_text)
        elif '[GENDER]' in line:
            for gender in GENDER_LIST:
                new_text = line.replace('[GENDER]', gender)
                sentences.append(new_text)
        elif '[INSURANCE]' in line:
            for insur in INSURANCE_LIST:
                new_text = line.replace('[INSURANCE]', insur)
                sentences.append(new_text)

    # Set up output directories
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR) 

    OUT_FILE = OUT_DIR / f'sentence_completion_results_{EXPERIMENT_CODE}.txt'

    # Loop through the models we are assessing
    for model_name in MODEL_LIST:
        if model_name[:4] != 'bert':  # i.e. is not one of the preloaded HuggingFace library models
            BERT_MODEL = '/scratch/gobi1/haoran/shared_data/BERT_DeBias/models/' + model_name
        else:
            BERT_MODEL = model_name
        
        print("Predicting words on ", BERT_MODEL)
        
        # Load pre-trained model with masked language model head
        model = BertForMaskedLM.from_pretrained(BERT_MODEL)
        model.eval()

        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
       
        top_words_results = []
        pred_sent_results = []

        if not USE_SLOW_DECODE:
            # collect top words and predicted sentences
            for sent in sentences:
                d, s = get_top_words_for_blank(sent, model, tokenizer, top_k)
                top_words_results.append(d)
                pred_sent_results.append(s)

        else:
            for sent in sentences:
                s = get_words_for_blank_slow_decode(sent, model, tokenizer)
                pred_sent_results.append(s)

        # write out results
        with open(OUT_FILE, 'a') as f:
            f.write('\n' + '>>>> ' + BERT_MODEL + " <<<<" + '\n')
            f.write('Use slow decode: ' + str(USE_SLOW_DECODE) + '\n')

            for i in range(len(pred_sent_results)):
                f.write(pred_sent_results[i] + "\n")
                if not USE_SLOW_DECODE:
                    f.write(str(top_words_results[i]) + "\n")
