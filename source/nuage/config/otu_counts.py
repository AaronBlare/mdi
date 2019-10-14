import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

class OTUCounts:

    def __init__(self,
                 otu_col_dict_T0,
                 otu_col_dict_T1,
                 subject_row_dict_T0,
                 subject_row_dict_T1,
                 normalized_T0,
                 normalized_T1,
                 rarified_T0,
                 rarified_T1,
                 raw_T0,
                 raw_T1,
                 ):
        self.otu_col_dict_T0 = otu_col_dict_T0
        self.otu_col_dict_T1 = otu_col_dict_T1
        self.subject_row_dict_T0 = subject_row_dict_T0
        self.subject_row_dict_T1 = subject_row_dict_T1
        self.normalized_T0 = normalized_T0
        self.normalized_T1 = normalized_T1
        self.rarified_T0 = rarified_T0
        self.rarified_T1 = rarified_T1
        self.raw_T0 = raw_T0
        self.raw_T1 = raw_T1


def load_otu_counts(fn, norm='none'):
    print('\n\n')
    f = open(fn)
    key_line = f.readline()
    keys = key_line.split('\t')
    keys[-1] = keys[-1].rstrip()
    keys = keys[3::]

    num_otus = len(keys)

    subj_id = 0
    time_id = 1
    type_id = 2

    times = []
    for line in tqdm(f):
        line_list = line.split('\t')
        line_list[-1] = line_list[-1].rstrip()
        time = line_list[time_id]
        times.append(time)
    f.close()

    number_of_T0 = int(times.count('T0') / 3)
    number_of_T1 = int(times.count('T1') / 3)

    normalized_T0 = np.zeros((number_of_T0, num_otus), dtype=np.float32)
    rarified_T0 = np.zeros((number_of_T0, num_otus), dtype=np.float32)
    raw_T0 = np.zeros((number_of_T0, num_otus), dtype=np.float32)

    normalized_T1 = np.zeros((number_of_T1, num_otus), dtype=np.float32)
    rarified_T1 = np.zeros((number_of_T1, num_otus), dtype=np.float32)
    raw_T1 = np.zeros((number_of_T1, num_otus), dtype=np.float32)

    subject_row_dict_T0 = {}
    subject_row_dict_T1 = {}
    curr_row_id_T0 = 0
    curr_row_id_T1 = 0
    subjects = []
    f = open(fn)
    f.readline()
    for line in tqdm(f):
        line_list = line.split('\t')
        line_list[-1] = line_list[-1].rstrip()

        subject = line_list[subj_id]
        subjects.append(subject)

        time = line_list[time_id]
        type = line_list[type_id]

        otus = line_list[3::]
        otus = np.float32(otus)
        if len(otus) != num_otus:
            raise ValueError('Wrong number of otus in row')

        if  type == 'NormalisedCount' and time == 'T0':
            normalized_T0[curr_row_id_T0] = otus
        elif type == 'RarifiedCount' and time == 'T0':
            rarified_T0[curr_row_id_T0] = otus
        elif type == 'RawCount' and time == 'T0':
            raw_T0[curr_row_id_T0] = otus
            subject_row_dict_T0[subject] = curr_row_id_T0
            curr_row_id_T0 += 1

        elif  type == 'NormalisedCount' and time == 'T1':
            normalized_T1[curr_row_id_T1] = otus
        elif type == 'RarifiedCount' and time == 'T1':
            rarified_T1[curr_row_id_T1] = otus
        elif type == 'RawCount' and time == 'T1':
            raw_T1[curr_row_id_T1] = otus
            subject_row_dict_T1[subject] = curr_row_id_T1
            curr_row_id_T1 += 1
    f.close()

    normalized_T0_nz = np.count_nonzero(normalized_T0, axis=0)
    normalized_T1_nz = np.count_nonzero(normalized_T1, axis=0)

    otu_col_dict_T0 = {}
    cols_to_del_T0 = []
    curr_T0_id = 0

    otu_col_dict_T1 = {}
    cols_to_del_T1 = []
    curr_T1_id = 0

    for key_id in range(0, len(keys)):

        missed_part_T0 = float(normalized_T0_nz[key_id]) / float(number_of_T0)
        if missed_part_T0 < 0.1:
            cols_to_del_T0.append(key_id)
        else:
            otu_col_dict_T0[keys[key_id]] = curr_T0_id
            curr_T0_id += 1

        missed_part_T1 = float(normalized_T1_nz[key_id]) / float(number_of_T1)
        if missed_part_T1 < 0.1:
            cols_to_del_T1.append(key_id)
        else:
            otu_col_dict_T1[keys[key_id]] = curr_T1_id
            curr_T1_id += 1

    print(f'Number of otu in T0: {len(otu_col_dict_T0)}')
    print(f'Number of otu in T1: {len(otu_col_dict_T1)}')

    normalized_T0 = np.delete(normalized_T0, cols_to_del_T0, axis=1)
    rarified_T0 = np.delete(rarified_T0, cols_to_del_T0, axis=1)
    raw_T0 = np.delete(raw_T0, cols_to_del_T0, axis=1)

    normalized_T1 = np.delete(normalized_T1, cols_to_del_T1, axis=1)
    rarified_T1 = np.delete(rarified_T1, cols_to_del_T1, axis=1)
    raw_T1 = np.delete(raw_T1, cols_to_del_T1, axis=1)

    if norm == 'normalize':
        normalized_T0 = preprocessing.normalize(normalized_T0, axis=0, norm='max')
        rarified_T0 = preprocessing.normalize(rarified_T0, axis=0, norm='max')
        raw_T0 = preprocessing.normalize(raw_T0, axis=0, norm='max')
        normalized_T1 = preprocessing.normalize(normalized_T1, axis=0, norm='max')
        rarified_T1 = preprocessing.normalize(rarified_T1, axis=0, norm='max')
        raw_T1 = preprocessing.normalize(raw_T1, axis=0, norm='max')
    elif norm == 'scale':
        normalized_T0 = preprocessing.scale(normalized_T0)
        rarified_T0 = preprocessing.scale(rarified_T0)
        raw_T0 = preprocessing.scale(raw_T0)
        normalized_T1 = preprocessing.scale(normalized_T1)
        rarified_T1 = preprocessing.scale(rarified_T1)
        raw_T1 = preprocessing.scale(raw_T1)

    subjects = list(set(subjects))
    print('Number of subjects with otus: ' + str(len(subjects)))
    print('Number of subjects with otus at T0: ' + str(len(subject_row_dict_T0)))
    print('Number of subjects with otus at T1: ' + str(len(subject_row_dict_T1)))
    common_subjects = list(set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys())))
    print('Number of subjects with otus at T0 and T1: ' + str(len(common_subjects)))

    otu_counts = OTUCounts(
        otu_col_dict_T0,
        otu_col_dict_T1,
        subject_row_dict_T0,
        subject_row_dict_T1,
        normalized_T0,
        normalized_T1,
        rarified_T0,
        rarified_T1,
        raw_T0,
        raw_T1
    )

    return otu_counts