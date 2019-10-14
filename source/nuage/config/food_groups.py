import numpy as np
from tqdm import tqdm
from sklearn import preprocessing


class FoodGroups:

    def __init__(self,
                 food_groups_col_dict,
                 subject_row_dicts,
                 food_groups_data_dict
                 ):
        self.food_groups_col_dict = food_groups_col_dict
        self.subject_row_dicts = subject_row_dicts
        self.food_groups_data_dict = food_groups_data_dict


def load_food_groups(fn, norm='none'):
    print('\n\n')
    f = open(fn)
    key_line = f.readline()
    keys = key_line.split('\t')
    keys[-1] = keys[-1].rstrip()
    f.readline()

    food_groups_col_dict = {}
    for key_id in range(2, len(keys)):
        if keys[key_id] in food_groups_col_dict:
            food_groups_col_dict[keys[key_id] + str(key_id)] = key_id - 2
        else:
            food_groups_col_dict[keys[key_id]] = key_id - 2
    f.readline()

    num_food_groups = len(food_groups_col_dict)

    subj_id = 0
    time_id = 1

    subjects = []
    times = []
    for line in tqdm(f):
        line_list = line.split('\t')
        line_list[-1] = line_list[-1].rstrip()
        subject = line_list[subj_id]
        subjects.append(subject)
        time = line_list[time_id]
        times.append(time)
    f.close()

    unique_times = list(set(times))
    number_of_times = len(unique_times)
    print(f'Number of unique times: {number_of_times}')
    num_points_in_times = {}
    subject_row_dicts = {}
    food_groups_data_dict = {}
    curr_row_dict = {}
    for time in unique_times:
        points_num = int(times.count(time))
        print(f'Number of points in time {time}: {points_num}')
        num_points_in_times[time] = points_num
        subject_row_dicts[time] = {}
        food_groups_data_dict[time] = np.zeros((points_num, num_food_groups), dtype=np.float32)
        curr_row_dict[time] = 0

    f = open(fn)
    f.readline()
    f.readline()
    f.readline()
    missed_ids  = set()
    for line in tqdm(f):
        line_list = line.split('\t')
        line_list[-1] = line_list[-1].rstrip()

        subject = line_list[subj_id]
        time = line_list[time_id]

        nutritions = line_list[2::]

        if nutritions.count('') < num_food_groups / 2:

            for n_id in range(0, len(nutritions)):
                if nutritions[n_id] == '':
                    missed_ids.add(n_id)
                    nutritions[n_id] = np.float32(0.0)
                elif nutritions[n_id] == '.':
                    nutritions[n_id] = np.float32(0.0)
                else:
                    nutritions[n_id] = np.float32(nutritions[n_id])

            if len(nutritions) != num_food_groups:
                raise ValueError('Wrong number of nutritions in row')

            subject_row_dicts[time][subject] = curr_row_dict[time]
            food_groups_data_dict[time][curr_row_dict[time]] = nutritions
            curr_row_dict[time] += 1

    f.close()

    subjects = list(set(subjects))
    print('Number of subjects with food_groups: ' + str(len(subjects)))
    for time in unique_times:
        print(f'Number of subjects with food_groups at {time}: {len(subject_row_dicts[time])}')
        if norm == 'normalize':
            food_groups_data_dict[time] = preprocessing.normalize(food_groups_data_dict[time], axis=0, norm='max')
        elif norm == 'scale':
            food_groups_data_dict[time] = preprocessing.scale(food_groups_data_dict[time])

    common_T0_T1_subjects = list(set(subject_row_dicts['T0'].keys()).intersection(set(subject_row_dicts['T1'].keys())))
    print('Number of subjects with food_groups at T0 and T1: ' + str(len(common_T0_T1_subjects)))

    food_groups = FoodGroups(
        food_groups_col_dict,
        subject_row_dicts,
        food_groups_data_dict
    )

    return food_groups
