#!/h/haoran/anaconda3/bin/python
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse
import spacy
import re
from heuristic_tokenize import sent_tokenize_rules

parser = argparse.ArgumentParser("Given a dataframe with a 'text' column, saves a dataframe to file, which is a copy of the input dataframe with 'sents_space' and 'toks' columns added on")
parser.add_argument("input_loc", help = "pickled dataframe with 'text' column", type=str)
parser.add_argument('output_loc', help = "path to output the dataframe", type=str)
parser.add_argument("model_path", help = 'folder with trained BERT model and tokenizer', type=str)
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertModel.from_pretrained(args.model_path)

df = pd.read_pickle(args.input_loc)

'''
Code taken from https://github.com/EmilyAlsentzer/clinicalBERT
'''
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    for c, i in enumerate(note_sections):
        note_sections[c] = re.sub('[0-9]+\.','' ,note_sections[c]) # remove '1.', '2.'
        note_sections[c] = re.sub('(-){2,}|_{2,}|={2,}','' ,note_sections[c]) # remove _____
        note_sections[c] = re.sub('dr\.','doctor' ,note_sections[c])
        note_sections[c] = re.sub('m\.d\.','md' ,note_sections[c])
    regex = '(\[\*\*[^*]*\*\*\])'
    processed_sections = [re.sub(regex, repl, i) for i in note_sections]
    processed_sections = [nlp(i.strip()) for i in processed_sections if i is not None and len(i.strip()) > 0]
    return(processed_sections) #list of spacy docs

def process_text(sent_text):
    if len(sent_text.strip()) > 0:
        sent_text = sent_text.replace('\n', ' ').strip()
        return sent_text
    return None

def get_sentences(doc):
    temp = []
    for i in doc.sents:
        s = process_text(i.string)
        if s is not None:
            temp.append(s)
    return temp

def process_note(note):
    sections = process_note_helper(note)
    sents = [j for i in sections for j in get_sentences(i)]
    sections = [i.text for i in sections]
    return (sents, sections)


'''
from https://github.com/wboag/synthID/blob/master/synth/synthid.py
'''
def is_date(string):
    string = string.lower()
    if re.search('^\d\d\d\d-\d\d?-\d\d?$', string): return string
    if re.search('^\d\d?-\d\d?$'         , string): return string
    if re.search('^\d\d\d\d$'            , string): return string
    if re.search('^\d\d?/\d\d\d\d$'      , string): return string
    if re.search('^\d-/\d\d\d\d$'        , string): return string[0]+string[2:]
    if re.search('january'               , string): return string
    if re.search('february'              , string): return string
    if re.search('march'                 , string): return string
    if re.search('april'                 , string): return string
    if re.search('may'                   , string): return string
    if re.search('june'                  , string): return string
    if re.search('july'                  , string): return string
    if re.search('august'                , string): return string
    if re.search('september'             , string): return string
    if re.search('october'               , string): return string
    if re.search('november'              , string): return string
    if re.search('december'              , string): return string
    if re.search('month'                 , string): return 'July'
    if re.search('year'                  , string): return '2012'
    if re.search('date range'            , string): return 'July - September'
    return False


def replace_deid(s):
    low_label = s.lower()
    date = is_date(low_label)
    if date or 'holiday' in low_label:
        label = 'PHIDATEPHI'

    elif 'hospital' in low_label:
        label = 'PHIHOSPITALPHI'

    elif ('location' in low_label
         or 'url ' in low_label
         or 'university' in low_label
         or 'address' in low_label
        or 'po box' in low_label
         or 'state' in low_label
         or 'country' in low_label
         or 'company' in low_label):
        label = 'PHILOCATIONPHI'


    elif ('name' in low_label
         or 'dictator info' in low_label
         or 'contact info' in low_label
         or 'attending info' in low_label):
        label = 'PHINAMEPHI'

    elif 'telephone' in low_label:
        label = 'PHICONTACTPHI'

    elif ('job number' in low_label
            or 'number' in low_label
            or 'numeric identifier' in low_label
            or re.search('^\d+$', low_label)
            or re.search('^[\d-]+$', low_label)
            or re.search('^[-\d/]+$', low_label)):
        label = 'PHINUMBERPHI'

    elif 'age over 90' in low_label:
        label = 'PHIAGEPHI'

    else:
        label = 'PHIOTHERPHI'

    return label

def repl(m):
    s = m.group(0)
    label = s[3:-3].strip()
    return replace_deid(label)

nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
nlp.add_pipe(sbd_component, before='parser')

df['sents'], df['sections'] = zip(*df.text.apply(process_note))
df['mod_text'] = df['sections'].apply(lambda x: '\n'.join(x))

tokens = []
for i in df.mod_text:
    tokens.append(tokenizer.tokenize(i))
df['toks'] = tokens
df['num_toks'] = df.toks.apply(len)
df = df[(df.num_toks > 0)]

def tokenize_sents(x):
    return [len(tokenizer.tokenize(i)) for i in x]

df['sent_toks_lens'] = df['sents'].apply(lambda x: tokenize_sents(x)) #length of each sent

# sentences could be composed of weird characters, that have length >= 1
# but when tokenized, they are dropped, resulting in empty sentences
def drop_bad_sents(x):
    i=0
    while i<len(x.sent_toks_lens):
        if x.sent_toks_lens[i] == 0:
            x.sent_toks_lens.pop(i)
            x.sents.pop(i)
            i -=1
        i+=1

#none of the sentences are empty
#assert(df.sent_toks_lens.apply(lambda x: sum([i == 0 for i in x])).sum() == 0)

df2 = df.loc[df.sent_toks_lens.apply(lambda x: sum([i == 0 for i in x])) > 0]
df2.apply(drop_bad_sents, axis = 1) #modifies sentence list in place

df.to_pickle(args.output_loc)
