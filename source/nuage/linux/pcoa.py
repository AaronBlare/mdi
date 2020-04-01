import os
from tqdm import tqdm
import numpy as np
import skbio
from skbio import read
from skbio.tree import TreeNode
from skbio.stats.ordination import pcoa
from linux.plot_pcoa import pcoa_plot


data_file_path = '/home/qiime2/Desktop/shared/nuage/linux/qiime/'
result_file_path = '/home/qiime2/Desktop/shared/nuage/linux/'
if not os.path.isdir(result_file_path):
    os.makedirs(result_file_path)

f = open(data_file_path + 'OTUcounts.tsv')
key_line = f.readline()
keys = key_line.split('\t')
keys[-1] = keys[-1].rstrip()
keys = keys[1::]
num_otus = len(keys)

subj_id = 0

subjects = []
for line in tqdm(f):
    line_list = line.split('\t')
    line_list[-1] = line_list[-1].rstrip()
    subjects.append(line_list[subj_id])
f.close()

otu_data_T0 = np.zeros((len(subjects), num_otus), dtype=np.float32)
subject_row_dict_T0 = {}
curr_row_id_T0 = 0
num_T0 = 0

f = open(data_file_path + 'OTUcounts.tsv')
f.readline()
for line in tqdm(f):
    line_list = line.split('\t')
    line_list[-1] = line_list[-1].rstrip()
    subject = line_list[subj_id]
    otus = line_list[1::]
    otus = np.float32(otus)
    otu_data_T0[curr_row_id_T0] = otus
    subject_row_dict_T0[subject] = curr_row_id_T0
    curr_row_id_T0 += 1
    num_T0 += 1
f.close()

otu_data_T0_nz = np.count_nonzero(otu_data_T0, axis=0)
otu_col_dict_T0 = {}
cols_to_del_T0 = []
curr_T0_id = 0

for key_id in range(0, len(keys)):

    presence_part_T0 = float(otu_data_T0_nz[key_id]) / float(num_T0)
    if presence_part_T0 < 0.1:
        cols_to_del_T0.append(key_id)
    else:
        otu_col_dict_T0[keys[key_id]] = curr_T0_id
        curr_T0_id += 1

data_T0 = np.delete(otu_data_T0, cols_to_del_T0, axis=1)

filename = result_file_path + 'tree.nwk'
tree = read(filename, format="newick", into=TreeNode)

dist_matrix = skbio.diversity.beta_diversity(metric='unweighted_unifrac',
                                             counts=data_T0,
                                             ids=list(subject_row_dict_T0.keys()),
                                             otu_ids=list(otu_col_dict_T0.keys()),
                                             tree=tree)
pcoa_results = pcoa(dist_matrix)
pcoa_plot(result_file_path + 'fig', pcoa_results, subjects)
