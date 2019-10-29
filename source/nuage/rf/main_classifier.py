from config.config import Config
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'

config = Config(data_file_path, figure_file_path)

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

otu_t0 = np.zeros((len(list(subject_row_dict_T0.keys())), len(list(config.otu_counts.otu_col_dict_T0.keys()))), dtype=np.float32)
otu_t1 = np.zeros((len(list(subject_row_dict_T1.keys())), len(list(config.otu_counts.otu_col_dict_T1.keys()))), dtype=np.float32)

common_otus = list(set(config.otu_counts.otu_col_dict_T0.keys()).intersection(set(config.otu_counts.otu_col_dict_T1.keys())))
common_otu_t0 = np.zeros((len(list(subject_row_dict_T0.keys())), len(common_otus)), dtype=np.float32)
common_otu_t1 = np.zeros((len(list(subject_row_dict_T1.keys())), len(common_otus)), dtype=np.float32)

otu_t0_id = 0
otu_t1_id = 0
otu_t0_col_dict = {}
otu_t1_col_dict = {}
for key in config.otu_counts.otu_col_dict_T0:
    otu_t0_col_dict[key] = otu_t0_id
    otu_t0[:, otu_t0_id] = config.otu_counts.normalized_T0[:, config.otu_counts.otu_col_dict_T0[key]]
    otu_t0_id += 1
for key in config.otu_counts.otu_col_dict_T1:
    otu_t1_col_dict[key] = otu_t1_id
    otu_t1[:, otu_t1_id] = config.otu_counts.normalized_T1[:, config.otu_counts.otu_col_dict_T1[key]]
    otu_t1_id += 1

otu_id = 0
otu_col_dict_common = {}
for key in config.otu_counts.otu_col_dict_T0:
    if key in config.otu_counts.otu_col_dict_T1:
        otu_col_dict_common[key] = otu_id
        common_otu_t0[:, otu_id] = config.otu_counts.normalized_T0[:, config.otu_counts.otu_col_dict_T0[key]]
        common_otu_t1[:, otu_id] = config.otu_counts.normalized_T1[:, config.otu_counts.otu_col_dict_T1[key]]
        otu_id += 1

subjects_t0 = []
controls_t0 = []
for person in list(subject_row_dict_T0.keys()):
    person_id = config.subject_info_T0['CODE'].index(person)
    if config.subject_info_T0['status'][person_id] == 'Subject':
        subjects_t0.append(person)
    if config.subject_info_T0['status'][person_id] == 'Control':
        controls_t0.append(person)
print('Number of Subjects at T0: ' + str(len(subjects_t0)))
print('Number of Controls at T0: ' + str(len(controls_t0)))

subjects_t1 = []
controls_t1 = []
for person in list(subject_row_dict_T1.keys()):
    person_id = config.subject_info_T0['CODE'].index(person)
    if config.subject_info_T0['status'][person_id] == 'Subject':
        subjects_t1.append(person)
    if config.subject_info_T0['status'][person_id] == 'Control':
        controls_t1.append(person)
print('Number of Subjects at T1: ' + str(len(subjects_t1)))
print('Number of Controls at T1: ' + str(len(controls_t1)))

adherence_key = 'compliance160'
adherence_key_t0_subject = 'adherence_t0_subject'
adherence_key_t0_control = 'adherence_t0_control'
adherence_key_t1_subject = 'adherence_t1_subject'
adherence_key_t1_control = 'adherence_t1_control'
adherence_dict = {adherence_key_t0_subject: [], adherence_key_t0_control: [],
                  adherence_key_t1_subject: [], adherence_key_t1_control: []}
adherence_diff_list_subject = []
adherence_diff_list_control = []

common_subjects = list(set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys())))
subjects = []
controls = []
subjects_wo_adherence = []
for code in common_subjects:
    index_t0 = config.subject_info_T0['CODE'].index(code)
    index_t1 = config.subject_info_T1['CODE'].index(code)
    curr_adherence_t0 = config.subject_info_T0[adherence_key][index_t0]
    curr_adherence_t1 = config.subject_info_T1[adherence_key][index_t1]

    if curr_adherence_t0 == '' or curr_adherence_t1 == '':
        subjects_wo_adherence.append(code)
        continue

    if config.subject_info_T0['status'][index_t0] == 'Subject':
        subjects.append(code)
        adherence_dict[adherence_key_t0_subject].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_subject].append(curr_adherence_t1)
        adherence_diff_list_subject.append(abs(curr_adherence_t0 - curr_adherence_t1))

    if config.subject_info_T0['status'][index_t0] == 'Control':
        controls.append(code)
        adherence_dict[adherence_key_t0_control].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_control].append(curr_adherence_t1)
        adherence_diff_list_control.append(abs(curr_adherence_t0 - curr_adherence_t1))

if len(subjects_wo_adherence) > 0:
    for elem in subjects_wo_adherence:
        common_subjects.remove(elem)

adherence_diff_tertiles_subject = pd.qcut(adherence_diff_list_subject, 3, labels=False)
adherence_diff_tertiles_control = pd.qcut(adherence_diff_list_control, 3, labels=False)

low_adherence_subject = []
low_adherence_subject_diff = []
low_adherence_control = []
low_adherence_control_diff = []
medium_adherence_subject = []
medium_adherence_subject_diff = []
medium_adherence_control = []
medium_adherence_control_diff = []
high_adherence_subject = []
high_adherence_subject_diff = []
high_adherence_control = []
high_adherence_control_diff = []

for index in range(0, len(subjects)):
    if adherence_diff_tertiles_subject[index] == 0:
        low_adherence_subject.append(subjects[index])
        low_adherence_subject_diff.append(adherence_diff_list_subject[index])
    elif adherence_diff_tertiles_subject[index] == 1:
        medium_adherence_subject.append(subjects[index])
        medium_adherence_subject_diff.append(adherence_diff_list_subject[index])
    elif adherence_diff_tertiles_subject[index] == 2:
        high_adherence_subject.append(subjects[index])

for index in range(0, len(controls)):
    if adherence_diff_tertiles_control[index] == 0:
        low_adherence_control.append(controls[index])
    elif adherence_diff_tertiles_control[index] == 1:
        medium_adherence_control.append(controls[index])
    elif adherence_diff_tertiles_control[index] == 2:
        high_adherence_control.append(controls[index])

otu_t0_subject = np.zeros((len(subjects_t0), len(list(config.otu_counts.otu_col_dict_T0.keys()))), dtype=np.float32)
otu_t0_control = np.zeros((len(controls_t0), len(list(config.otu_counts.otu_col_dict_T0.keys()))), dtype=np.float32)
otu_t1_subject = np.zeros((len(subjects_t1), len(list(config.otu_counts.otu_col_dict_T1.keys()))), dtype=np.float32)
otu_t1_control = np.zeros((len(controls_t1), len(list(config.otu_counts.otu_col_dict_T1.keys()))), dtype=np.float32)

common_otu_t0_subject = np.zeros((len(subjects_t0), len(common_otus)), dtype=np.float32)
common_otu_t0_control = np.zeros((len(controls_t0), len(common_otus)), dtype=np.float32)
common_otu_t1_subject = np.zeros((len(subjects_t1), len(common_otus)), dtype=np.float32)
common_otu_t1_control = np.zeros((len(controls_t1), len(common_otus)), dtype=np.float32)

for sub_id, sub in enumerate(subjects_t0):
    curr_otu_t0 = otu_t0[subject_row_dict_T0[sub], :]
    curr_otu_t0_common = common_otu_t0[subject_row_dict_T0[sub], :]

    otu_t0_subject[sub_id, :] = curr_otu_t0
    common_otu_t0_subject[sub_id, :] = curr_otu_t0_common

for sub_id, sub in enumerate(controls_t0):
    curr_otu_t0 = otu_t0[subject_row_dict_T0[sub], :]
    curr_otu_t0_common = common_otu_t0[subject_row_dict_T0[sub], :]

    otu_t0_control[sub_id, :] = curr_otu_t0
    common_otu_t0_control[sub_id, :] = curr_otu_t0_common

for sub_id, sub in enumerate(subjects_t1):
    curr_otu_t1 = otu_t1[subject_row_dict_T1[sub], :]
    curr_otu_t1_common = common_otu_t1[subject_row_dict_T1[sub], :]

    otu_t1_subject[sub_id, :] = curr_otu_t1
    common_otu_t1_subject[sub_id, :] = curr_otu_t1_common

for sub_id, sub in enumerate(controls_t1):
    curr_otu_t1 = otu_t1[subject_row_dict_T1[sub], :]
    curr_otu_t1_common = common_otu_t1[subject_row_dict_T1[sub], :]

    otu_t1_control[sub_id, :] = curr_otu_t1
    common_otu_t1_control[sub_id, :] = curr_otu_t1_common

otu_t0_subject_df = pd.DataFrame(otu_t0_subject, subjects_t0, list(otu_t0_col_dict.keys()))
otu_t0_control_df = pd.DataFrame(otu_t0_control, controls_t0, list(otu_t0_col_dict.keys()))
otu_t1_subject_df = pd.DataFrame(otu_t1_subject, subjects_t1, list(otu_t1_col_dict.keys()))
otu_t1_control_df = pd.DataFrame(otu_t1_control, controls_t1, list(otu_t1_col_dict.keys()))

common_otu_t0_subject_df = pd.DataFrame(common_otu_t0_subject, subjects_t0, list(otu_col_dict_common.keys()))
common_otu_t0_control_df = pd.DataFrame(common_otu_t0_control, controls_t0, list(otu_col_dict_common.keys()))
common_otu_t1_subject_df = pd.DataFrame(common_otu_t1_subject, subjects_t1, list(otu_col_dict_common.keys()))
common_otu_t1_control_df = pd.DataFrame(common_otu_t1_control, controls_t1, list(otu_col_dict_common.keys()))

subj_control_t1_df = otu_t1_subject_df.append(otu_t1_control_df)
classes_subj_control_t1 = ['Subject',] * len(subjects_t1) + ['Control',] * len(controls_t1)
factor = pd.factorize(classes_subj_control_t1)
y = factor[0]
clf_subj_control_t1 = RandomForestClassifier(n_estimators=500)
output_subj_control_t1 = cross_validate(clf_subj_control_t1, subj_control_t1_df, y, cv=5, scoring='accuracy', return_estimator=True)
accuracy = np.mean(output_subj_control_t1['test_score'])
print('Accuracy Subject T1 vs Control T1: ' + str(accuracy))

subj_control_t0_df = otu_t0_subject_df.append(otu_t0_control_df)
classes_subj_control_t0 = ['Subject',] * len(subjects_t0) + ['Control',] * len(controls_t0)
factor = pd.factorize(classes_subj_control_t0)
y = factor[0]
clf_subj_control_t0 = RandomForestClassifier(n_estimators=500)
output_subj_control_t0 = cross_validate(clf_subj_control_t0, subj_control_t0_df, y, cv=5, scoring='accuracy', return_estimator=True)
accuracy = np.mean(output_subj_control_t0['test_score'])
print('Accuracy Subject T0 vs Control T0: ' + str(accuracy))

subj_t0_t1_df = common_otu_t0_subject_df.append(common_otu_t1_subject_df)
classes_subj_t0_t1 = ['T0',] * len(subjects_t0) + ['T1',] * len(subjects_t1)
factor = pd.factorize(classes_subj_t0_t1)
y = factor[0]
clf_subj_t0_t1 = RandomForestClassifier(n_estimators=500)
output_subj_t0_t1 = cross_validate(clf_subj_t0_t1, subj_t0_t1_df, y, cv=5, scoring='accuracy', return_estimator=True)
accuracy = np.mean(output_subj_t0_t1['test_score'])
print('Accuracy Subject T0 vs Subject T1: ' + str(accuracy))

control_t0_t1_df = common_otu_t0_control_df.append(common_otu_t1_control_df)
classes_control_t0_t1 = ['T0',] * len(controls_t0) + ['T1',] * len(controls_t1)
factor = pd.factorize(classes_control_t0_t1)
y = factor[0]
clf_control_t0_t1 = RandomForestClassifier(n_estimators=500)
output_control_t0_t1 = cross_validate(clf_control_t0_t1, control_t0_t1_df, y, cv=5, scoring='accuracy', return_estimator=True)
accuracy = np.mean(output_control_t0_t1['test_score'])
print('Accuracy Control T0 vs Control T1: ' + str(accuracy))
