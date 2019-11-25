import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import Constants
import sys
from pathlib import Path

output_folder = Path(sys.argv[1])
output_folder.mkdir(parents = True, exist_ok = True)

conn = psycopg2.connect('dbname=mimic user=haoran host=mimic password=password')

pats = pd.read_sql_query('''
select subject_id, gender, dob, dod from mimiciii.patients
''', conn)

n_splits = 12
pats = pats.sample(frac = 1, random_state = 42).reset_index(drop = True)
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
for c,i in enumerate(kf.split(pats, groups = pats.gender)):
    pats.loc[i[1], 'fold'] = str(c)

adm = pd.read_sql_query('''
select subject_id, hadm_id, insurance, language,
religion, ethnicity,
admittime, deathtime, dischtime,
HOSPITAL_EXPIRE_FLAG, DISCHARGE_LOCATION,
diagnosis as adm_diag
from mimiciii.admissions
''', conn)

df = pd.merge(pats, adm, on='subject_id', how = 'inner')

def merge_death(row):
    if not(pd.isnull(row.deathtime)):
        return row.deathtime
    else:
        return row.dod
df['dod_merged'] = df.apply(merge_death, axis = 1)


notes = pd.read_sql_query('''
select category, chartdate, charttime, hadm_id, row_id as note_id, text from mimiciii.noteevents
where iserror is null
''', conn)

# drop all outpatients. They only have a subject_id, so can't link back to insurance or other fields
notes = notes[~(pd.isnull(notes['hadm_id']))]

df = pd.merge(left = notes, right = df, on='hadm_id', how = 'left')

df.ethnicity.fillna(value = 'UNKNOWN/NOT SPECIFIED', inplace = True)

others_set = set()
def cleanField(string):
    mappings = {'HISPANIC OR LATINO': 'HISPANIC/LATINO',
                'BLACK/AFRICAN AMERICAN': 'BLACK',
                'UNABLE TO OBTAIN':'UNKNOWN/NOT SPECIFIED',
               'PATIENT DECLINED TO ANSWER': 'UNKNOWN/NOT SPECIFIED'}
    bases = ['WHITE', 'UNKNOWN/NOT SPECIFIED', 'BLACK', 'HISPANIC/LATINO',
            'OTHER', 'ASIAN']

    if string in bases:
        return string
    elif string in mappings:
        return mappings[string]
    else:
        for i in bases:
            if i in string:
                return i
        others_set.add(string)
        return 'OTHER'

df['ethnicity_to_use'] = df['ethnicity'].apply(cleanField)

df = df[df.chartdate >= df.dob]

ages = []
for i in range(df.shape[0]):
    ages.append((df.chartdate.iloc[i] - df.dob.iloc[i]).days/365.24)
df['age'] = ages

df.loc[(df.category == 'Discharge summary') |
       (df.category == 'Echo') |
       (df.category == 'ECG'), 'fold'] = 'NA'

icds = (pd.read_sql_query('select * from mimiciii.diagnoses_icd', conn)
        .groupby('hadm_id')
        .agg({'icd9_code': lambda x: list(x.values)})
        .reset_index())

df = pd.merge(left = df, right = icds, on = 'hadm_id')

def map_lang(x):
    if x == 'ENGL':
        return 'English'
    if pd.isnull(x):
        return 'Missing'
    return 'Other'
df['language_to_use'] = df['language'].apply(map_lang)


for i in Constants.groups:
    assert(i['name'] in df.columns), i['name']

acuities = pd.read_sql_query('''
select * from (
select a.subject_id, a.hadm_id, a.icustay_id, a.oasis, a.oasis_prob, b.sofa from
(mimiciii.oasis a
natural join mimiciii.sofa b )) ab
natural join
(select subject_id, hadm_id, icustay_id, sapsii, sapsii_prob from
mimiciii.sapsii) c
''', conn)

icustays = pd.read_sql_query('''
select subject_id, hadm_id, icustay_id, intime, outtime
from mimiciii.icustays
''', conn).set_index(['subject_id','hadm_id'])

def fill_icustay(row):
    opts = icustays.loc[[row['subject_id'],row['hadm_id']]]
    if pd.isnull(row['charttime']):
        charttime = row['chartdate'] + pd.Timedelta(days = 2)
    else:
        charttime = row['charttime']
    stay = opts[(opts['intime'] <= charttime)].sort_values(by = 'intime', ascending = True)

    if len(stay) == 0:
        return None
        #print(row['subject_id'], row['hadm_id'], row['category'])
    return stay.iloc[-1]['icustay_id']

df['icustay_id'] = df[df.category.isin(['Discharge summary','Physician ','Nursing','Nursing/other'])].apply(fill_icustay, axis = 1)

df = pd.merge(df, acuities.drop(columns = ['subject_id','hadm_id']), on = 'icustay_id', how = 'left')
df.loc[df.age >= 90, 'age'] = 91.4

df.to_pickle(output_folder / "df_raw.pkl")
