# code is a mess since it's mostly exported from a jupyter notebook
import os
import pandas as pd
import Constants
from sklearn.model_selection import KFold
from readers import PhenotypingReader, InHospitalMortalityReader
import yaml
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--processed_df', type=Path, required=True)
parser.add_argument('--mimic_benchmark_dir', type = Path, required = True)
parser.add_argument('--output_dir', type = Path, required = True)
args = parser.parse_args()

def preprocessing(row):
    '''
    Input: a list of tokens
    Output: a list of string, with each string having MAX_SEQ_LEN-2 tokens
    Uses a sliding window approach, with the window sliding (SLIDING_DIST tokens) each time
    '''
    n = int(len(row.toks)/Constants.SLIDING_DIST)
    seqs = []
    if n == 0: # note shorter than SLIDING_DIST tokens
        seqs.append(' '.join(row.toks))
    else:
        for j in range(min(n, Constants.MAX_NUM_SEQ)):
            seqs.append(' '.join(row.toks[j*Constants.SLIDING_DIST:(j+1)*Constants.SLIDING_DIST+(Constants.MAX_SEQ_LEN - Constants.SLIDING_DIST-2)]))
    return seqs

df = pd.read_pickle(args.processed_df.resolve())
df['seqs'] = df.apply(preprocessing, axis = 1)
df['num_seqs'] = df.seqs.apply(len)
assert(df.seqs.apply(lambda x: any([len(i)== 0 for i in x])).sum() == 0)

df['note_id'] = df['note_id'].astype(str)
df = df[~pd.isnull(df['oasis'])]
MAX_AGG_SEQUENCE_LENGTH = Constants.MAX_AGG_SEQUENCE_LEN

root_folder = args.mimic_benchmark_dir/'root/'
other_features = ['age', 'oasis', 'oasis_prob', 'sofa', 'sapsii', 'sapsii_prob']

'''
In-Hospital Mortality
Using the first 48 hours of patient information within their ICU stay, predict whether or not the patient will die in hospital.
Subjects/targets are extracted by MIMIC-Benchmarks script. Their script only extracts numeric data, while we want to use only notes.
Their script also defines a new time scale, so that t=0 is when the patient first enters the ICU.

What we do is:
- Using the MIMIC-Benchmarks InHospitalMortalityReader, read in each patient, to get the target.
We know the period of interest will be 0-48 hours, where 0 is the intime to the ICU.
- For each patient, we obtain their icustay_id from the episode file in their data folder
- We obtain their (hadm_id, intime) from all_stays.csv using their icustay_id
- With this information, along with the 48 hour period length, we can index
into df to obtain a set of note_ids corresponding to that period
- To construct a training set for each individual, we take sequences from the
last k notes, until the patient runs out of notes, or we reach the max_agg_sequence_length.
- We only use sequences from the following note types: Nursing, Nursing/Other,
Physician
- We take the last k notes, because they are more likely to be informative of
the target, compared to the first notes
- We assign a new ID for this aggregated note, which is a combination of their
subject ID and episode number
'''


train_reader = InHospitalMortalityReader(dataset_dir=args.mimic_benchmark_dir/'in-hospital-mortality' / 'train')
test_reader = InHospitalMortalityReader(dataset_dir=args.mimic_benchmark_dir/'in-hospital-mortality' / 'test')
all_stays = pd.read_csv(os.path.join(root_folder, 'all_stays.csv'), parse_dates = ['INTIME']).set_index('ICUSTAY_ID')

def read_patient(name, period_length, allowed_types, eps = 0.001, dtype = 'train', return_intime = False):
    # given a file name, retrieve all notes from t=-eps to period_length+eps
    subj_id = int(name.split('_')[0])
    stay = pd.read_csv(os.path.join(root_folder, dtype, str(subj_id), name.split('_')[1]+'.csv'))
    assert(stay.shape[0] == 1)
    row = stay.iloc[0]

    icuid = row['Icustay']
    hadm_id = all_stays.loc[icuid]['HADM_ID']
    intime = all_stays.loc[icuid]['INTIME']
    result = df[(df['subject_id'] == subj_id) & (df['hadm_id'] == hadm_id)
       & (df['charttime'] >= intime) & (df['charttime'] < intime+pd.Timedelta(hours = period_length + eps))
             & (df['category'].isin(allowed_types))]
    if return_intime:
        return (intime, result)
    else:
        return result

def agg_notes(notes, first = False, intime = None, timeDiff = pd.Timedelta(hours = 48)):
    notes = notes.sort_values(by = 'charttime', ascending = False)
    seqs = []
    note_ids = []
    if first:
        note_to_take = None
        firstgood = notes[notes.category.isin(['Nursing', 'Physician '])]
        if firstgood.shape[0] > 0 and (firstgood.iloc[0]['charttime'] - intime) <= timeDiff:
            note_to_take = firstgood.iloc[0]
        elif (notes.iloc[0]['charttime'] - intime) <= timeDiff:
            note_to_take = notes.iloc[0]
        if note_to_take is not None:
            seqs = note_to_take['seqs']
            note_ids.append(note_to_take['note_id'])

    else:
        for idx, row in notes.iterrows():
            if len(seqs) + row.num_seqs <= MAX_AGG_SEQUENCE_LENGTH:
                seqs = row.seqs + seqs
                note_ids = [row.note_id] + note_ids
    return {**{
        'insurance': notes.iloc[0]['insurance'],
        'gender': notes.iloc[0]['gender'],
        'ethnicity_to_use': notes.iloc[0]['ethnicity_to_use'],
        'language_to_use': notes.iloc[0]['language_to_use'],
        'subject_id': notes.iloc[0]['subject_id'],
        'hadm_id': notes.iloc[0]['hadm_id'],
        'seqs': seqs,
        'note_ids': note_ids,
        'num_seqs': len(seqs),
    }, **{i: notes.iloc[0][i] for i in other_features}}

temp = []
for i in range(train_reader.get_number_of_examples()):
    ex = train_reader.read_example(i)
    notes = read_patient(ex['name'], 48, ['Nursing', 'Physician ', 'Nursing/other'])
    if len(notes) > 0: #no notes of interest within first 48 hours
        dic = agg_notes(notes)
        dic['inhosp_mort'] = ex['y']
        dic['note_id'] = ''.join(ex['name'].split('_')[:2]) + 'a'
        dic['fold'] = 'train'
        temp.append(dic)

for i in range(test_reader.get_number_of_examples()):
    ex = test_reader.read_example(i)
    notes = read_patient(ex['name'], 48, ['Nursing', 'Physician ', 'Nursing/other'], dtype = 'test')
    if len(notes) > 0: #no notes of interest within first 48 hours
        dic = agg_notes(notes)
        dic['inhosp_mort'] = ex['y']
        dic['note_id'] = ''.join(ex['name'].split('_')[:2])+ 'a'
        dic['fold'] = 'test'
        temp.append(dic)
t2 = pd.DataFrame(temp)
# split training set into folds, stratify by inhosp_mort
subjects = t2.loc[t2['fold'] != 'test',['subject_id', 'inhosp_mort']].groupby('subject_id').first().reset_index()
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
for c,j in enumerate(kf.split(subjects, groups = subjects['inhosp_mort'])):
    for k in j[1]:
        t2.loc[t2['subject_id'] == subjects.loc[k]['subject_id'], 'fold'] = str(c+1)
t2.to_pickle(args.output_dir / 'inhosp_mort')


'''
Phenotyping using all patient notes
- Using the MIMIC-Benchmarks PhenotypingReader, read in each patient, to get
the targets and the period length (which is the length of stay). We know the period of interest will be 0 to los + $\epsilon$,
where 0 is the intime to the ICU, and $\epsilon$ is a small number (so discharge notes are included).
- We obtain (hadm_id, intime) usin the same method above
- With this information, along with the los + $\epsilon$ hour period length, we
can index into df to obtain a set of note_ids corresponding to that period
- We construct sequences using the last k notes, in the same manner as above.
- We only use sequences from the following note types: Nursing, Nursing/Other,
Physician, Discharge Summary
- We also add in the following targets, aggregated from the specific
phenotypes: Any acute, Any chronic, Any disease
'''

with open('../icd9_codes.yml', 'r') as f:
    ccs = pd.DataFrame.from_dict(yaml.load(f)).T

target_names = list(pd.read_csv(os.path.join(root_folder, 'phenotype_labels.csv')).columns)
acutes = [i for i in target_names if ccs.loc[i, 'type'] == 'acute']
chronics = [i for i in target_names if ccs.loc[i, 'type'] == 'chronic']
train_reader = PhenotypingReader(dataset_dir=args.mimic_benchmark_dir/'phenotyping' / 'train')
test_reader = PhenotypingReader(dataset_dir=args.mimic_benchmark_dir/'phenotyping' / 'test')
temp = []
def has_any(dic, keys):
    return any([dic[i] == 1 for i in keys])

for i in range(train_reader.get_number_of_examples()):
    ex = train_reader.read_example(i)
    notes = read_patient(ex['name'], float(ex['t']), ['Nursing', 'Physician ', 'Nursing/other', 'Discharge summary'])
    if len(notes) > 0:
        dic = agg_notes(notes)
        for tar, y in zip(target_names, ex['y']):
            dic[tar] = y
        dic['any_acute'] = has_any(dic, acutes)
        dic['any_chronic'] = has_any(dic, chronics)
        dic['any_disease'] = has_any(dic, target_names)

        dic['note_id'] = ''.join(ex['name'].split('_')[:2]) + 'b'
        dic['fold'] = 'train'
        temp.append(dic)

for i in range(test_reader.get_number_of_examples()):
    ex = test_reader.read_example(i)
    notes = read_patient(ex['name'], float(ex['t']), ['Nursing', 'Physician ', 'Nursing/other', 'Discharge summary'], dtype = 'test')
    if len(notes) > 0:
        dic = agg_notes(notes)
        for tar, y in zip(target_names, ex['y']):
            dic[tar] = y
        dic['any_acute'] = has_any(dic, acutes)
        dic['any_chronic'] = has_any(dic, chronics)
        dic['any_disease'] = has_any(dic, target_names)

        dic['note_id'] = ''.join(ex['name'].split('_')[:2]) + 'b'
        dic['fold'] = 'test'
        temp.append(dic)

cols = target_names + ['any_chronic', 'any_acute', 'any_disease']
t3 = pd.DataFrame(temp)
subjects = t3.loc[t3['fold'] != 'test',['subject_id', 'any_disease']].groupby('subject_id').first().reset_index()
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
for c,j in enumerate(kf.split(subjects, groups = subjects['any_disease'])):
    for k in j[1]:
        t3.loc[t3['subject_id'] == subjects.loc[k]['subject_id'], 'fold'] = str(c+1)

t3.to_pickle(args.output_dir / 'phenotype_all')

'''
Phenotyping using the first patient note
- Using the MIMIC-Benchmarks PhenotypingReader, read in each patient, to get
the targets and the period length (which is the length of stay). We know the period of interest will be 0 to los + $\epsilon$,
where 0 is the intime to the ICU, and $\epsilon$ is a small number (so discharge notes are included).
- We obtain (hadm_id, intime) usin the same method above
- With this information, along with the los + $\epsilon$ hour period length, we
can index into df. We take the first nursing or physician note within the first 48 hours of a person's stay.
If this does not exist, we take the first nursing/other note within the first 48 hours.
- If they do not have a nursing note within 48 hours of their intime, the
patient is dropped.
'''

train_reader = PhenotypingReader(dataset_dir=args.mimic_benchmark_dir/'phenotyping' / 'train')
test_reader = PhenotypingReader(dataset_dir=args.mimic_benchmark_dir/'phenotyping' / 'test')
temp = []
for i in range(train_reader.get_number_of_examples()):
    ex = train_reader.read_example(i)
    intime, notes = read_patient(ex['name'], float(ex['t']), ['Nursing', 'Physician ', 'Nursing/other'], return_intime = True)
    if len(notes) > 0:
        dic = agg_notes(notes, first = True, intime = intime)
        if len(dic['seqs']) == 0:
            continue
        for tar, y in zip(target_names, ex['y']):
            dic[tar] = y
        dic['any_acute'] = has_any(dic, acutes)
        dic['any_chronic'] = has_any(dic, chronics)
        dic['any_disease'] = has_any(dic, target_names)

        dic['note_id'] = dic['note_ids'][0]
        del dic['note_ids']
        dic['fold'] = 'train'
        temp.append(dic)

for i in range(test_reader.get_number_of_examples()):
    ex = test_reader.read_example(i)
    intime, notes = read_patient(ex['name'], float(ex['t']), ['Nursing', 'Physician ', 'Nursing/other'], dtype = 'test', return_intime = True)
    if len(notes) > 0:
        dic = agg_notes(notes, first = True, intime = intime)
        if len(dic['seqs']) == 0:
            continue
        for tar, y in zip(target_names, ex['y']):
            dic[tar] = y
        dic['any_acute'] = has_any(dic, acutes)
        dic['any_chronic'] = has_any(dic, chronics)
        dic['any_disease'] = has_any(dic, target_names)

        dic['note_id'] = dic['note_ids'][0]
        del dic['note_ids']
        dic['fold'] = 'test'
        temp.append(dic)
t4 = pd.DataFrame(temp)
t4 = pd.merge(t4, df[['note_id', 'category']], on = 'note_id', how = 'left')
subjects = t4.loc[t4['fold'] != 'test',['subject_id', 'any_disease']].groupby('subject_id').first().reset_index()
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
for c,j in enumerate(kf.split(subjects, groups = subjects['any_disease'])):
    for k in j[1]:
        t4.loc[t4['subject_id'] == subjects.loc[k]['subject_id'], 'fold'] = str(c+1)
t4.to_pickle(args.output_dir / 'phenotype_first')
