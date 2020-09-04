import os
from tqdm import tqdm
import numpy as np
import skbio
from skbio import read
from skbio.tree import TreeNode
from skbio.stats.ordination import pcoa
from linux.plot_pcoa import pcoa_plot
from skbio.stats.distance import DistanceMatrix

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

f = open(data_file_path + 'Vat_SatT0.csv')
key_line = f.readline()
subjects = []
for line in tqdm(f):
    line_list = line.split(',')
    if len(line_list) > 1:
        subjects.append(line_list[1])

t0_t1_subjects = list(set(subjects_T0).intersection(set(subjects_T1)))
common_subjects = list(set(t0_t1_subjects).intersection(set(subjects)))
common_subjects.sort()

otu_data_T0 = np.zeros((len(common_subjects), num_otus), dtype=np.float32)
subject_row_dict_T0 = {}
curr_row_id_T0 = 0
num_T0 = 0

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

filename = result_file_path + 'qiime/tree_qiime.nwk'
tree = read(filename, format="newick", into=TreeNode)

dists = np.zeros((len(common_subjects), len(common_subjects)))
for i in tqdm(range(0, len(common_subjects))):
    for j in range(0, len(common_subjects)):
        if i >= j:
            continue
        else:
            index_1 = subject_row_dict_T0[common_subjects[i]]
            counts_1 = data_T0[index_1, :]
            index_2 = subject_row_dict_T0[common_subjects[j]]
            counts_2 = data_T0[index_2, :]
            curr_dist = skbio.diversity.beta.weighted_unifrac(v_counts=counts_1,
                                                              u_counts=counts_2,
                                                              otu_ids=list(otu_col_dict_T0.keys()),
                                                              tree=tree,
                                                              normalized=False,
                                                              validate=True)
            dists[i, j] = curr_dist
            dists[j, i] = curr_dist

dist_matrix = DistanceMatrix(dists)
# dist_matrix = skbio.diversity.beta_diversity(metric='unweighted_unifrac',
#                                              counts=data_T0,
#                                              ids=list(subject_row_dict_T0.keys()),
#                                              otu_ids=list(otu_col_dict_T0.keys()),
#                                              tree=tree)
pcoa_results = pcoa(dist_matrix)
pcoa_plot(result_file_path + 'fig/mafft_tree', pcoa_results, common_subjects)
