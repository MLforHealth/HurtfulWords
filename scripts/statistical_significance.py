import sys
import numpy as np
from scipy import stats
import pandas as pd
import ipdb

for fpath in sys.argv[1:]:
    df = pd.read_csv(fpath, sep='\t')
    categories = np.unique(df.categories)

    for cat in categories:
        tmp = df[df.categories == cat]
        m = tmp[tmp.demographic == 'male'].log_probs
        f = tmp[tmp.demographic == 'female'].log_probs

        m_mean = np.mean(m)
        f_mean = np.mean(f)
        
        ks = stats.ks_2samp(m, f)
        wilcoxon = stats.ranksums(m, f)
        ttest = stats.ttest_ind(m, f)

        print('****', fpath, '****')
        print('****', cat, '****')
        print('male mean,\t', m_mean)
        print('female mean,\t', f_mean)
        
        print(">>KS TEST<<")
        print("D test statistic,\t", ks[0])
        print("p-value,\t", ks[1])

        print(">>WILCOXON RANK-SUMS<<")
        print("Test statistic,\t", wilcoxon[0])
        print("p-value,\t", wilcoxon[1])

        print(">>T-TEST<<")
        print("Test statistic,\t", ttest[0])
        print("p-value,\t", ttest[1])

        print()
