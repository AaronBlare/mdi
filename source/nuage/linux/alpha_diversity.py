import os
from tqdm import tqdm
import numpy as np
import skbio
import pandas as pd

data_file_path = '/home/qiime2/Desktop/shared/nuage/'
result_file_path = '/home/qiime2/Desktop/shared/nuage/linux/'
if not os.path.isdir(result_file_path):
    os.makedirs(result_file_path)

f = open(data_file_path + 'OTUcounts.tsv')
key_line = f.readline()
keys = key_line.split('\t')
keys[-1] = keys[-1].rstrip()
keys = keys[3::]
num_otus = len(keys)

subj_id = 0
time_id = 1
type_id = 2

subjects_T0 = []
subjects_T1 = []
for line in tqdm(f):
    line_list = line.split('\t')
    line_list[-1] = line_list[-1].rstrip()
    time = line_list[time_id]
    if time == 'T0':
        subjects_T0.append(line_list[subj_id])
    elif time == 'T1':
        subjects_T1.append(line_list[subj_id])
f.close()

common_subjects = list(set(subjects_T0).intersection(set(subjects_T1)))
otu_data_T0 = np.zeros((len(common_subjects), num_otus), dtype=np.float32)

subject_row_dict_T0 = {}
curr_row_id_T0 = 0

f = open(data_file_path + 'OTUcounts.tsv')
f.readline()
for line in tqdm(f):
    line_list = line.split('\t')
    line_list[-1] = line_list[-1].rstrip()

    subject = line_list[subj_id]
    if subject in common_subjects:
        time = line_list[time_id]
        type = line_list[type_id]

        otus = line_list[3::]
        otus = np.float32(otus)

        if type == 'RarifiedCount' and time == 'T0':
            otu_data_T0[curr_row_id_T0] = otus
            subject_row_dict_T0[subject] = curr_row_id_T0
            curr_row_id_T0 += 1
f.close()

otu_col_dict_T0 = {}
curr_T0_id = 0

for key_id in range(0, len(keys)):
    otu_col_dict_T0[keys[key_id]] = curr_T0_id
    curr_T0_id += 1

groups_file_name = 'RDP_classified_TaxaTable.tsv'
f = open(data_file_path + groups_file_name)
key_line = f.readline()
otu_group_dict = {}
otu_group_names = []
for line in f:
    otu_data = line.split('\t')
    otu_name = otu_data[0]
    otu_group = otu_data[-2]
    if otu_group not in otu_group_names:
        otu_group_names.append(otu_group)
    otu_group_dict[otu_name] = otu_group
f.close()

otu_col_dict = otu_col_dict_T0
subject_row_dict = subject_row_dict_T0
otu_counts = otu_data_T0

subject_group_dict = {key: np.zeros(len(otu_group_names), dtype=int) for key in subject_row_dict}
for subject_id in range(0, len(list(subject_row_dict.keys()))):
    curr_subject = list(subject_row_dict.keys())[subject_id]
    for otu_id in range(0, len(list(otu_col_dict.keys()))):
        curr_otu_name = list(otu_col_dict.keys())[otu_id]
        curr_otu_val = otu_counts[subject_id, otu_id]
        if curr_otu_val > 0.0:
            curr_otu_group = otu_group_dict[curr_otu_name]
            curr_index = otu_group_names.index(curr_otu_group)
            subject_group_dict[curr_subject][curr_index] += 1

subject_diversity_dict = {key: {'Shannon': -1.0, 'Fisher': -1.0, 'Number': -1.0, 'Simpson': -1.0, 'Chao1': -1.0} for key
                          in subject_row_dict}
for subject in subject_diversity_dict:
    shannon_diversity = skbio.diversity.alpha.shannon(subject_group_dict[subject], base=np.e)
    fisher_diversity = skbio.diversity.alpha.fisher_alpha(subject_group_dict[subject])
    number_diversity = np.sum(subject_group_dict[subject])
    simpson_diversity = skbio.diversity.alpha.simpson(subject_group_dict[subject])
    chao1_diversity = skbio.diversity.alpha.chao1(subject_group_dict[subject])
    subject_diversity_dict[subject]['Shannon'] = shannon_diversity
    subject_diversity_dict[subject]['Fisher'] = fisher_diversity
    subject_diversity_dict[subject]['Number'] = number_diversity
    subject_diversity_dict[subject]['Simpson'] = simpson_diversity
    subject_diversity_dict[subject]['Chao1'] = chao1_diversity

df = pd.DataFrame.from_dict(data=subject_diversity_dict)
df_transposed = df.T
df_transposed.index.name = 'ID'
df_transposed.reset_index(inplace=True)
df_transposed.to_excel(result_file_path + 'alpha_diversity.xlsx', header=True)
