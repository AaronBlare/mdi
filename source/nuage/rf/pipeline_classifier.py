import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def pipeline_classifier(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otus = config.get_common_otus()
    common_otu_t0, common_otu_t1 = config.separate_common_otus()

    config.separate_data('otu_counts', 'T0')
    otu_t0 = config.curr_data
    otu_t0_col_dict = config.curr_col_dict

    config.separate_data('otu_counts', 'T1')
    otu_t1 = config.curr_data
    otu_t1_col_dict = config.curr_col_dict

    status_key = 'status'

    metadata_status_t0, obs_dict_status_t0 = config.get_target_subject_dicts(subject_row_dict_T0.keys(), [status_key],
                                                                             'T0')
    metadata_status_t1, obs_dict_status_t1 = config.get_target_subject_dicts(subject_row_dict_T1.keys(), [status_key],
                                                                             'T1')

    print('Number of Subjects at T0: ' + str(len(obs_dict_status_t0[status_key]['Subject'])))
    print('Number of Controls at T0: ' + str(len(obs_dict_status_t0[status_key]['Control'])))
    print('Number of Subjects at T1: ' + str(len(obs_dict_status_t1[status_key]['Subject'])))
    print('Number of Controls at T1: ' + str(len(obs_dict_status_t1[status_key]['Control'])))

    adherence_key = 'compliance160'
    common_subjects = config.get_common_subjects()
    metadata_ad_t0, obs_dict_ad_t0 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
    metadata_ad_t1, obs_dict_ad_t1 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

    adherence_key_t0_subject = 't0_subject'
    adherence_key_t0_control = 't0_control'
    adherence_key_t1_subject = 't1_subject'
    adherence_key_t1_control = 't1_control'
    adherence_dict = {adherence_key_t0_subject: [], adherence_key_t0_control: [],
                      adherence_key_t1_subject: [], adherence_key_t1_control: []}
    adherence_diff_list_subject = []
    adherence_diff_list_control = []

    subjects_common = []
    controls_common = []
    subjects_wo_adherence = []
    for code in common_subjects:
        curr_adherence_t0 = metadata_ad_t0[code][adherence_key]
        curr_adherence_t1 = metadata_ad_t1[code][adherence_key]
        curr_status = metadata_status_t0[code][status_key]

        if curr_adherence_t0 == '' or curr_adherence_t1 == '':
            subjects_wo_adherence.append(code)
            continue

        if curr_status == 'Subject':
            subjects_common.append(code)
            adherence_dict[adherence_key_t0_subject].append(curr_adherence_t0)
            adherence_dict[adherence_key_t1_subject].append(curr_adherence_t1)
            adherence_diff_list_subject.append(abs(curr_adherence_t0 - curr_adherence_t1))

        if curr_status == 'Control':
            controls_common.append(code)
            adherence_dict[adherence_key_t0_control].append(curr_adherence_t0)
            adherence_dict[adherence_key_t1_control].append(curr_adherence_t1)
            adherence_diff_list_control.append(abs(curr_adherence_t0 - curr_adherence_t1))

    if len(subjects_wo_adherence) > 0:
        for elem in subjects_wo_adherence:
            common_subjects.remove(elem)

    diff_percentile_subject_val = 20
    diff_percentiles_control_val = 5
    t0_percentiles_control_val = 5
    t1_percentiles_control_val = 5

    adherence_diff_percentiles_subject = pd.qcut(adherence_diff_list_subject, diff_percentile_subject_val, labels=False)
    adherence_diff_percentiles_control = pd.qcut(adherence_diff_list_control, diff_percentiles_control_val,
                                                 labels=False)
    adherence_t0_percentiles_control = pd.qcut(adherence_dict[adherence_key_t0_control], t0_percentiles_control_val,
                                               labels=False)
    adherence_t1_percentiles_control = pd.qcut(adherence_dict[adherence_key_t1_control], t1_percentiles_control_val,
                                               labels=False)

    low_adherence_subject = []
    low_adherence_subject_diff = []
    low_adherence_control = []
    low_adherence_control_diff = []
    high_adherence_subject = []
    high_adherence_subject_diff = []
    high_adherence_control = []
    high_adherence_control_diff = []

    low_adherence_low_diff_control = []

    for index in range(0, len(subjects_common)):
        if adherence_diff_percentiles_subject[index] == 0:
            low_adherence_subject.append(subjects_common[index])
            low_adherence_subject_diff.append(adherence_diff_list_subject[index])
        elif adherence_diff_percentiles_subject[index] == diff_percentile_subject_val - 1:
            high_adherence_subject.append(subjects_common[index])
            high_adherence_subject_diff.append(adherence_diff_list_subject[index])

    for index in range(0, len(controls_common)):
        if adherence_diff_percentiles_control[index] == 0:
            low_adherence_control.append(controls_common[index])
            low_adherence_control_diff.append(adherence_diff_list_control[index])
            if adherence_t0_percentiles_control[index] == 0 or adherence_t1_percentiles_control[index] == 0:
                low_adherence_low_diff_control.append(controls_common[index])
        elif adherence_diff_percentiles_control[index] == diff_percentiles_control_val - 1:
            high_adherence_control.append(controls_common[index])
            high_adherence_control_diff.append(adherence_diff_list_control[index])

    otu_t0_subject_df = configure_dataframe(otu_t0, obs_dict_status_t0[status_key]['Subject'],
                                            list(otu_t0_col_dict.keys()), subject_row_dict_T0)
    otu_t0_control_df = configure_dataframe(otu_t0, obs_dict_status_t0[status_key]['Control'],
                                            list(otu_t0_col_dict.keys()), subject_row_dict_T0)
    otu_t1_subject_df = configure_dataframe(otu_t1, obs_dict_status_t1[status_key]['Subject'],
                                            list(otu_t1_col_dict.keys()), subject_row_dict_T1)
    otu_t1_control_df = configure_dataframe(otu_t1, obs_dict_status_t1[status_key]['Control'],
                                            list(otu_t1_col_dict.keys()), subject_row_dict_T1)

    common_otu_t0_subject_df = configure_dataframe(common_otu_t0, obs_dict_status_t0[status_key]['Subject'],
                                                   common_otus, subject_row_dict_T0)
    common_otu_t0_control_df = configure_dataframe(common_otu_t0, obs_dict_status_t0[status_key]['Control'],
                                                   common_otus, subject_row_dict_T0)
    common_otu_t1_subject_df = configure_dataframe(common_otu_t1, obs_dict_status_t1[status_key]['Subject'],
                                                   common_otus, subject_row_dict_T1)
    common_otu_t1_control_df = configure_dataframe(common_otu_t1, obs_dict_status_t1[status_key]['Control'],
                                                   common_otus, subject_row_dict_T1)

    otu_high_adherence_diff_subject_df = configure_dataframe(common_otu_t1, high_adherence_subject,
                                                             common_otus, subject_row_dict_T1)
    otu_low_adherence_diff_control_df = configure_dataframe(common_otu_t1, low_adherence_low_diff_control,
                                                            common_otus, subject_row_dict_T1)

    high_subj_low_control_df = otu_high_adherence_diff_subject_df.append(otu_low_adherence_diff_control_df)
    classes_subj_control_t1 = ['Subject', ] * len(high_adherence_subject) + \
                              ['Control', ] * len(low_adherence_low_diff_control)

    accuracy = run_classifier(high_subj_low_control_df, classes_subj_control_t1)
    print('Accuracy High Adherence Diff Subject T1 vs Low Adherence Diff Control T1: ' + str(accuracy))

    subj_control_t1_df = otu_t1_subject_df.append(otu_t1_control_df)
    classes_subj_control_t1 = ['Subject', ] * len(obs_dict_status_t1[status_key]['Subject']) + \
                              ['Control', ] * len(obs_dict_status_t1[status_key]['Control'])
    accuracy = run_classifier(subj_control_t1_df, classes_subj_control_t1)
    print('Accuracy Subject T1 vs Control T1: ' + str(accuracy))

    subj_control_t0_df = otu_t0_subject_df.append(otu_t0_control_df)
    classes_subj_control_t0 = ['Subject', ] * len(obs_dict_status_t0[status_key]['Subject']) + \
                              ['Control', ] * len(obs_dict_status_t0[status_key]['Control'])
    accuracy = run_classifier(subj_control_t0_df, classes_subj_control_t0)
    print('Accuracy Subject T0 vs Control T0: ' + str(accuracy))

    subj_t0_t1_df = common_otu_t0_subject_df.append(common_otu_t1_subject_df)
    classes_subj_t0_t1 = ['T0', ] * len(obs_dict_status_t0[status_key]['Subject']) + \
                         ['T1', ] * len(obs_dict_status_t1[status_key]['Subject'])
    accuracy = run_classifier(subj_t0_t1_df, classes_subj_t0_t1)
    print('Accuracy Subject T0 vs Subject T1: ' + str(accuracy))

    control_t0_t1_df = common_otu_t0_control_df.append(common_otu_t1_control_df)
    classes_control_t0_t1 = ['T0', ] * len(obs_dict_status_t0[status_key]['Control']) + \
                            ['T1', ] * len(obs_dict_status_t1[status_key]['Control'])
    accuracy = run_classifier(control_t0_t1_df, classes_control_t0_t1)
    print('Accuracy Control T0 vs Control T1: ' + str(accuracy))


def configure_dataframe(data, sub_list, var_list, subject_row_dict):
    df_array = np.zeros((len(sub_list), len(var_list)), dtype=np.float32)

    for sub_id, sub in enumerate(sub_list):
        curr_otu_t0 = data[subject_row_dict[sub], :]
        df_array[sub_id, :] = curr_otu_t0
    result_df = pd.DataFrame(df_array, sub_list, var_list)

    return result_df


def run_classifier(data, classes):

    factor = pd.factorize(classes)
    y = factor[0]
    clf = RandomForestClassifier(n_estimators=500)
    output = cross_validate(clf, data, y, cv=5, scoring='accuracy', return_estimator=True)
    accuracy = np.mean(output['test_score'])

    return accuracy
