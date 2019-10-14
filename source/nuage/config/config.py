from config.otu_counts import load_otu_counts
from config.nutrition import load_nutrition
from config.food_groups import load_food_groups
from config.subjects import load_subject_info, T0_T1_subject_separation

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
