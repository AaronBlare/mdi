import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def pipeline_classifier(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otus = config.get_common_otus()
    common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

    config.separate_data('otu_counts', 'T0')
    otu_t0 = config.curr_data
    otu_t0_col_dict = config.curr_col_dict

    config.separate_data('otu_counts', 'T1')
    otu_t1 = config.curr_data
    otu_t1_col_dict = config.curr_col_dict

    common_subjects = config.get_common_subjects_with_adherence()
    otu_counts_delta = config.get_otu_counts_delta()

    status_key = 'status'

    metadata_status_t0, obs_dict_status_t0 = config.get_target_subject_dicts(list(subject_row_dict_T0.keys()),
                                                                             [status_key], 'T0')
    metadata_status_t1, obs_dict_status_t1 = config.get_target_subject_dicts(list(subject_row_dict_T1.keys()),
                                                                             [status_key], 'T1')

    print('Number of Subjects at T0: ' + str(len(obs_dict_status_t0[status_key]['Subject'])))
    print('Number of Controls at T0: ' + str(len(obs_dict_status_t0[status_key]['Control'])))
    print('Number of Subjects at T1: ' + str(len(obs_dict_status_t1[status_key]['Subject'])))
    print('Number of Controls at T1: ' + str(len(obs_dict_status_t1[status_key]['Control'])))

    adherence_diff, subject_row_adherence_dict = config.get_adherence_diff(common_subjects)

    subjects_common, controls_common = separate_status(metadata_status_t0, common_subjects)

    diff_percentile_val = 4
    adherence_diff_percentiles = pd.qcut(adherence_diff, diff_percentile_val, labels=False, retbins=True)

    high_adherence = []
    low_adherence = []

    for index in range(0, len(common_subjects)):
        if adherence_diff_percentiles[0][index] == diff_percentile_val - 1:
            high_adherence.append(common_subjects[index])
        if adherence_diff_percentiles[0][index] == 0:
            low_adherence.append(common_subjects[index])

    adherence_diff_subject_df = configure_dataframe(np.array(adherence_diff)[:, np.newaxis], subjects_common,
                                                    ['adherence'], subject_row_adherence_dict)
    adherence_diff_control_df = configure_dataframe(np.array(adherence_diff)[:, np.newaxis], controls_common,
                                                    ['adherence'], subject_row_adherence_dict)

    otu_high_adherence_diff_df = configure_dataframe(otu_counts_delta.data, high_adherence,
                                                     common_otus, otu_counts_delta.subject_row_dict)
    otu_low_adherence_diff_df = configure_dataframe(otu_counts_delta.data, low_adherence,
                                                    common_otus, otu_counts_delta.subject_row_dict)

    otu_t0_subject_df = configure_dataframe(otu_t0, obs_dict_status_t0[status_key]['Subject'],
                                            list(otu_t0_col_dict.keys()), subject_row_dict_T0)
    otu_t0_control_df = configure_dataframe(otu_t0, obs_dict_status_t0[status_key]['Control'],
                                            list(otu_t0_col_dict.keys()), subject_row_dict_T0)
    otu_t1_subject_df = configure_dataframe(otu_t1, obs_dict_status_t1[status_key]['Subject'],
                                            list(otu_t1_col_dict.keys()), subject_row_dict_T1)
    otu_t1_control_df = configure_dataframe(otu_t1, obs_dict_status_t1[status_key]['Control'],
                                            list(otu_t1_col_dict.keys()), subject_row_dict_T1)

    otu_t0_df = configure_dataframe(common_otu_t0, common_subjects, common_otus, subject_row_dict_T0)
    otu_t1_df = configure_dataframe(common_otu_t1, common_subjects, common_otus, subject_row_dict_T1)

    otu_subject_df = configure_dataframe(otu_counts_delta.data, subjects_common,
                                         common_otus, otu_counts_delta.subject_row_dict)
    otu_control_df = configure_dataframe(otu_counts_delta.data, controls_common,
                                         common_otus, otu_counts_delta.subject_row_dict)

    common_otu_t0_subject_df = configure_dataframe(common_otu_t0, obs_dict_status_t0[status_key]['Subject'],
                                                   common_otus, subject_row_dict_T0)
    common_otu_t0_control_df = configure_dataframe(common_otu_t0, obs_dict_status_t0[status_key]['Control'],
                                                   common_otus, subject_row_dict_T0)
    common_otu_t1_subject_df = configure_dataframe(common_otu_t1, obs_dict_status_t1[status_key]['Subject'],
                                                   common_otus, subject_row_dict_T1)
    common_otu_t1_control_df = configure_dataframe(common_otu_t1, obs_dict_status_t1[status_key]['Control'],
                                                   common_otus, subject_row_dict_T1)

    otu_high_adherence_t1_df = configure_dataframe(common_otu_t1, high_adherence,
                                                   common_otus, subject_row_dict_T1)
    otu_low_adherence_t1_df = configure_dataframe(common_otu_t1, low_adherence,
                                                  common_otus, subject_row_dict_T1)

    adherence_diff_df = adherence_diff_subject_df.append(adherence_diff_control_df)
    classes_adherence_diff = ['Subject', ] * len(subjects_common) + \
                             ['Control', ] * len(controls_common)

    accuracy = run_classifier(adherence_diff_df, classes_adherence_diff)
    print('Accuracy Adherence Diff Subject vs Adherence Diff Control: ' + str(accuracy))

    high_low_t1_df = otu_high_adherence_t1_df.append(otu_low_adherence_t1_df)
    classes_high_low_t1 = ['High', ] * len(high_adherence) + \
                          ['Low', ] * len(low_adherence)

    accuracy = run_classifier(high_low_t1_df, classes_high_low_t1)
    print('Accuracy High Adherence T1 vs Low Adherence T1: ' + str(accuracy))

    high_low_diff_df = otu_high_adherence_diff_df.append(otu_low_adherence_diff_df)
    classes_high_low_diff = ['High', ] * len(high_adherence) + \
                            ['Low', ] * len(low_adherence)

    accuracy = run_classifier(high_low_diff_df, classes_high_low_diff)
    print('Accuracy High Adherence Diff vs Low Adherence Diff: ' + str(accuracy))

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

    t0_t1_df = otu_t0_df.append(otu_t1_df)
    classes_t0_t1 = ['T0', ] * len(common_subjects) + \
                    ['T1', ] * len(common_subjects)
    accuracy = run_classifier(t0_t1_df, classes_t0_t1)
    print('Accuracy T0 vs T1: ' + str(accuracy))

    subject_control_df = otu_subject_df.append(otu_control_df)
    classes_subject_control = ['Subject', ] * len(subjects_common) + \
                              ['Control', ] * len(controls_common)
    accuracy = run_classifier(subject_control_df, classes_subject_control)
    print('Accuracy Subject vs Control: ' + str(accuracy))


def separate_status(metadata_status, common_subjects):
    subjects = []
    controls = []
    for code in common_subjects:
        curr_status = metadata_status[code]['status']
        if curr_status == 'Subject':
            subjects.append(code)
        elif curr_status == 'Control':
            controls.append(code)
    return subjects, controls


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
