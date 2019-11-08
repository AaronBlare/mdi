from config.otu_counts import load_otu_counts, OTUCountsDelta
from config.nutrition import load_nutrition
from config.food_groups import load_food_groups
from config.subjects import load_subject_info, T0_T1_subject_separation
import numpy as np

class Config:

    def __init__(self,
                 path_in,
                 path_out
                 ):

        self.path_in = path_in
        self.path_out = path_out

        self.subject_info = load_subject_info(self.path_in + '/' + 'correct_subject_info.tsv')
        self.subject_info_T0, self.subject_info_T1 = T0_T1_subject_separation(self.subject_info)

        self.food_groups = load_food_groups(self.path_in + '/' + 'food_groups_long.tsv', norm='normalize')
        self.nutrition = load_nutrition(self.path_in + '/' + 'nutrition_long.tsv', norm='normalize')
        self.otu_counts = load_otu_counts(self.path_in + '/' + 'OTUcounts.tsv', norm='none')

    def get_common_subjects(self):
        food_subj_row_dict_T0 = self.food_groups.subject_row_dicts['T0']
        food_subj_row_dict_T1 = self.food_groups.subject_row_dicts['T1']

        nut_subj_row_dict_T0 =  self.nutrition.subject_row_dicts['T0']
        nut_subj_row_dict_T1 =  self.nutrition.subject_row_dicts['T1']

        subject_row_dict_T0 =  self.otu_counts.subject_row_dict_T0
        subject_row_dict_T1 =  self.otu_counts.subject_row_dict_T1

        common_subjects_food = set(food_subj_row_dict_T0.keys()).intersection(set(food_subj_row_dict_T1.keys()))
        common_subjects_nutrition = set(nut_subj_row_dict_T0.keys()).intersection(set(nut_subj_row_dict_T1.keys()))
        common_subjects_otu = set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys()))

        common_subjects = list(common_subjects_otu.intersection(common_subjects_nutrition).intersection(common_subjects_food))
        print(f'\nNumber of common subjects: {len(common_subjects)}')

        return common_subjects

    def get_common_subjects_with_adherence(self):
        common_subjects = self.get_common_subjects()

        adherence_key = 'compliance160'
        metadata_ad_t0, obs_dict_ad_t0 = self.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
        metadata_ad_t1, obs_dict_ad_t1 = self.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

        subjects_wo_adherence = []
        for code in common_subjects:
            curr_adherence_t0 = metadata_ad_t0[code][adherence_key]
            curr_adherence_t1 = metadata_ad_t1[code][adherence_key]
            if curr_adherence_t0 == '' or curr_adherence_t1 == '':
                subjects_wo_adherence.append(code)
                continue

        if len(subjects_wo_adherence) > 0:
            for elem in subjects_wo_adherence:
                common_subjects.remove(elem)

        print(f'\nNumber of common subjects with adherence: {len(common_subjects)}')

        return common_subjects

    def get_target_subject_dicts(self, subjects, keys, time='T0'):
        metadata = {}

        obs_dict = {}
        for key in keys:
            obs_dict[key] = {}

        if time == 'T0':
            subject_dict = self.subject_info_T0
        elif time == 'T1':
            subject_dict = self.subject_info_T1
        else:
            raise ValueError('Wrong time segment')

        for code in subjects:
            index = subject_dict['CODE'].index(code)

            curr_dict = {}
            for key in keys:
                curr_val = subject_dict[key][index]
                curr_dict[key] = curr_val

                if curr_val in obs_dict[key]:
                    obs_dict[key][curr_val].append(code)
                else:
                    obs_dict[key][curr_val] = [code]

            metadata[code] = curr_dict

        return metadata, obs_dict

    def separate_data(self, data_type, time):

        if data_type == 'otu_counts':
            if time == 'T0':
                col_dict = self.otu_counts.otu_col_dict_T0
                row_dict = self.otu_counts.subject_row_dict_T0
                data = self.otu_counts.normalized_T0
            elif time == 'T1':
                col_dict = self.otu_counts.otu_col_dict_T1
                row_dict = self.otu_counts.subject_row_dict_T1
                data = self.otu_counts.normalized_T1
            else:
                raise ValueError('Wrong time segment')
        else:
            if time in ['T0', 'T1']:
                if data_type == 'nutrition':
                    col_dict = self.nutrition.nutrition_col_dict
                    row_dict = self.nutrition.subject_row_dicts[time]
                    data = self.nutrition.nutrition_data_dict[time]
                elif data_type == 'food_groups':
                    col_dict = self.food_groups.food_groups_col_dict
                    row_dict = self.food_groups.subject_row_dicts[time]
                    data = self.food_groups.food_groups_data_dict[time]
                else:
                    raise ValueError('Wrong data_type')
            else:
                raise ValueError('Wrong time segment')

        self.curr_col_dict = col_dict
        self.curr_raw_dict = row_dict
        self.curr_data = data

    def separate_common_otus(self):
        num_rows_t0 = len(list(self.otu_counts.subject_row_dict_T0))
        num_rows_t1 = len(list(self.otu_counts.subject_row_dict_T1))
        num_cols = len(self.get_common_otus())

        common_otu_t0 = np.zeros((num_rows_t0, num_cols), dtype=np.float32)
        common_otu_t1 = np.zeros((num_rows_t1, num_cols), dtype=np.float32)

        otu_id = 0
        common_otu_col_dict = {}
        for key in self.otu_counts.otu_col_dict_T0:
            if key in self.otu_counts.otu_col_dict_T1:
                common_otu_col_dict[key] = otu_id
                common_otu_t0[:, otu_id] = self.otu_counts.normalized_T0[:, self.otu_counts.otu_col_dict_T0[key]]
                common_otu_t1[:, otu_id] = self.otu_counts.normalized_T1[:, self.otu_counts.otu_col_dict_T1[key]]
                otu_id += 1
        return common_otu_t0, common_otu_t1, common_otu_col_dict

    def get_common_otus(self):
        common_otus = []
        for key in self.otu_counts.otu_col_dict_T0:
            if key in self.otu_counts.otu_col_dict_T1:
                common_otus.append(key)
        return common_otus

    def get_adherence_diff(self, common_subjects):
        adherence_key = 'compliance160'
        metadata_ad_t0, obs_dict_ad_t0 = self.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
        metadata_ad_t1, obs_dict_ad_t1 = self.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

        adherence_diff = []
        subject_row_dict = {}
        for sub_id, sub in enumerate(common_subjects):
            curr_adherence_t0 = metadata_ad_t0[sub][adherence_key]
            curr_adherence_t1 = metadata_ad_t1[sub][adherence_key]
            adherence_diff.append(curr_adherence_t1 - curr_adherence_t0)
            subject_row_dict[sub] = sub_id
        return adherence_diff, subject_row_dict

    def get_otu_counts_delta(self):

        common_subjects = self.get_common_subjects_with_adherence()

        subject_row_dict_T0 = self.otu_counts.subject_row_dict_T0
        subject_row_dict_T1 = self.otu_counts.subject_row_dict_T1

        common_otu_t0, common_otu_t1, common_otu_col_dict = self.separate_common_otus()

        data = np.zeros((len(common_subjects), common_otu_t0.shape[1]), dtype=np.float32)
        subject_row_dict = {}

        for sub_id, sub in enumerate(common_subjects):

            subject_row_dict[sub] = sub_id
            curr_otu_t0 = common_otu_t0[subject_row_dict_T0[sub], :]
            curr_otu_t1 = common_otu_t1[subject_row_dict_T1[sub], :]
            curr_otu_diff = np.zeros((1, len(curr_otu_t0)), dtype=np.float32)

            for otu_id in range(0, len(curr_otu_t0)):
                curr_otu_diff[0, otu_id] = curr_otu_t1[otu_id] - curr_otu_t0[otu_id]
            data[sub_id, :] = curr_otu_diff

        otu_counts_delta = OTUCountsDelta(common_otu_col_dict, subject_row_dict, data)

        return otu_counts_delta
