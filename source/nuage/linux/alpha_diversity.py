import os
import numpy as np
from config.config import Config

data_file_path = '/home/qiime2/Desktop/shared/nuage/'
result_file_path = '/home/qiime2/Desktop/shared/nuage/linux/'
if not os.path.isdir(result_file_path):
    os.makedirs(result_file_path)
config = Config(data_file_path, result_file_path)

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

otu_col_dict = config.otu_counts.otu_col_dict_T0
subject_row_dict = config.otu_counts.subject_row_dict_T0
otu_counts = config.otu_counts.raw_T0

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
