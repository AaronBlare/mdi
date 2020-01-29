import os
from tqdm import tqdm
import numpy as np
import skbio
from skbio import TreeNode
from io import StringIO


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

result_file = open('/home/qiime2/Desktop/shared/nuage/linux/OTU_newick.txt', 'r')
tree_from_file = result_file.readline().rstrip()
result_file.close()

tree = TreeNode.read(StringIO(tree_from_file))

beta_diversity = skbio.diversity.beta_diversity(metric='unweighted_unifrac',
                                                counts=otu_data_T0,
                                                ids=list(subject_row_dict_T0.keys()),
                                                otu_ids=list(otu_col_dict_T0.keys()),
                                                tree=tree)
