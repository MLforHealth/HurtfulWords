from collections import defaultdict
groups = [
{'name': 'age', 'type' : 'ordinal', 'bins': list(enumerate([
[0, 10],
[10, 20],
[20, 30],
[30, 40],
[40, 50],
[50, 60],
[60, 70],
[70, 80],
[80, 90],
[90, 100000]
]))},
{'name': 'ethnicity_to_use', 'type': 'categorical'},
{'name': 'gender', 'type' : 'categorical'},
{'name': 'insurance', 'type': 'categorical'},
{'name': 'language_to_use', 'type': 'categorical'}
]

mapping={
    'gender':{
        'M': 0,
        'F': 1
    },
    'ethnicity_to_use': {
        'WHITE': 0,
        'BLACK': 1,
        'ASIAN': 2,
        'HISPANIC/LATINO': 3,
        'OTHER': 4,
        'UNKNOWN/NOT SPECIFIED': 5
    },
    'insurance': {
        'Medicare': 0,
        'Private': 1,
        'Medicaid': 2,
        'Government': 3,
        'Self Pay': 4
    },
    'language_to_use': {
        'English': 0,
        'Other': 1,
        'Missing' : 2
    }
}

newmapping={
    'gender':{
        'M': 0,
        'F': 1
    },
    'ethnicity_to_use': {
        'WHITE': 0,
        'BLACK': 1,
        'ASIAN': 2,
        'HISPANIC/LATINO': 3,
        'OTHER': 4,
        'UNKNOWN/NOT SPECIFIED': 5
    },
    'insurance': {
        'Medicare': 0,
        'Private': 1,
        'Medicaid': 2,
        'Government': 2,
        'Self Pay':3
    },
    'language_to_use': {
        'English': 0,
        'Other': 1,
        'Missing' : 2
    }
}

drop_groups = {
    'ethnicity_to_use': ['UNKNOWN/NOT SPECIFIED'],
    'language_to_use': ['Missing'],
    'insurance': ['Self Pay']
}
drop_groups = defaultdict(list, drop_groups)

for i in groups:
    if i['type'] == 'categorical':
        assert(i['name'] in mapping)

MAX_SEQ_LEN = 512
SLIDING_DIST = 256 #how much to slide the window by at each step during fine tuning
MAX_NUM_SEQ = 10 #maximum number of sequences to use during fine tuning from a single note
MAX_AGG_SEQUENCE_LEN = 30 # max number of notes to aggregate for finetuning
