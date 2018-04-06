import pandas as pd
import numpy as np

def read_crispr_data(filename):
	data = pd.read_csv(filename)
	return data

data = read_crispr_data('../GenomeCRISPR_full05112017.csv')

# check how many unique seqs
len(data['sequence'])
# 1048575
len(set(data['sequence']))
# 239895

data.groupby(['sequence', 'start']).mean()
# 255289

# get avg and std of effects
m = data.groupby(['sequence']).mean()['effect']
s = data.groupby(['sequence']).std()['effect']

# get number of seqs with only 1 repeat or same effect twice
len(np.where(s == 0)[0])
# 3531

np.max(s)
# 10

# distribution of effect classes
data.groupby(['effect']).count()['sequence']