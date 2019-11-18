from rf.plot import plot_box, plot_hist, plot_scatter, plot_heatmap, plot_random_forest
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from scipy.stats import spearmanr


def pipeline_regressor(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otus = config.get_common_otus()

    common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

    adherence_key = 'compliance160'
    adherence_key_t0 = 'adherence_t0'
    adherence_key_t1 = 'adherence_t1'
    adherence_dict = {adherence_key_t0: [], adherence_key_t1: []}

    common_subjects = config.get_common_subjects_with_adherence()
    metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
    metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

    for code in common_subjects:
        curr_adherence_t0 = metadata_t0[code][adherence_key]
        curr_adherence_t1 = metadata_t1[code][adherence_key]

        adherence_dict[adherence_key_t0].append(curr_adherence_t0 * 100.0 / 160.0)
        adherence_dict[adherence_key_t1].append(curr_adherence_t1 * 100.0 / 160.0)

    otu_t0 = np.zeros((len(common_subjects), len(common_otus)), dtype=np.float32)
    otu_t1 = np.zeros((len(common_subjects), len(common_otus)), dtype=np.float32)

    for sub_id, sub in tqdm(enumerate(common_subjects)):
        curr_otu_t0 = common_otu_t0[subject_row_dict_T0[sub], :]
        curr_otu_t1 = common_otu_t1[subject_row_dict_T1[sub], :]

        otu_t0[sub_id, :] = curr_otu_t0
        otu_t1[sub_id, :] = curr_otu_t1

    otu_t0_df = pd.DataFrame(otu_t0, common_subjects, list(config.otu_col_dict.keys()))
    otu_t1_df = pd.DataFrame(otu_t1, common_subjects, list(config.otu_col_dict.keys()))

    top_features_t0, top_features_imp_t0 = run_regressor(config, otu_t0_df, adherence_dict, adherence_key_t0, 'T0')
    top_features_t1, top_features_imp_t1 = run_regressor(config, otu_t1_df, adherence_dict, adherence_key_t1, 'T1')

    top_features_merged = list(set(top_features_t0 + top_features_t1))
    top_features_intersection_imp = []
    for feature_id in range(0, len(top_features_merged)):
        name = top_features_merged[feature_id]
        curr_imp = []
        if name in top_features_t0:
            index_t0 = top_features_t0.index(name)
            imp_t0 = top_features_imp_t0[index_t0]
            curr_imp.append(imp_t0)
        if name in top_features_t1:
            index_t1 = top_features_t1.index(name)
            imp_t1 = top_features_imp_t1[index_t1]
            curr_imp.append(imp_t1)
        if len(curr_imp) == 1:
            top_features_intersection_imp.append(curr_imp[0])
        else:
            top_features_intersection_imp.append(np.mean(curr_imp))

    top_features_paper = []
    otu_file = config.path_in + '/' + 'otu_random_forest.txt'
    f = open(otu_file)
    for line in f:
        top_features_paper.append(line.replace(' \n', ''))
    f.close()

    top_features_common_with_paper = list(set(top_features_merged).intersection(set(top_features_paper)))
    print('Number of common OTUs: ', str(len(top_features_common_with_paper)))

    new_df_t0 = otu_t0_df[top_features_merged].copy()
    new_df_t1 = otu_t1_df[top_features_merged].copy()
    new_df = new_df_t0.append(new_df_t1)
    new_adherence = adherence_dict[adherence_key_t0] + adherence_dict[adherence_key_t1]

    corr_coeffs = []
    for i in range(0, len(top_features_merged)):
        corr_coeff, p_val = spearmanr(list(new_df.iloc[:, i]), new_adherence)
        corr_coeffs.append(corr_coeff)

    data, names = map(list, zip(*sorted(zip(corr_coeffs, top_features_merged), reverse=False)))
    plot_heatmap(data, list(range(1, len(top_features_merged))), config.path_out)

    name_list = []
    bact_list = []
    fn_otus_info = config.path_in + '/Spingo_classified_TaxaTable.tsv'
    f = open(fn_otus_info)
    f.readline()
    for line in f:
        line = line.split('\t')
        name = line[0]
        bact = line[-2]
        name_list.append(name)
        bact_list.append(bact)
    f.close()

    for name_id in range(0, len(top_features_merged)):
        name = top_features_merged[name_id]
        index = name_list.index(name)
        top_features_merged[name_id] = name + '_' + bact_list[index]

    diet_positive_names = []
    diet_positive_imp = []
    diet_negative_names = []
    diet_negative_imp = []

    for coeff_id in range(0, len(corr_coeffs)):
        curr_coeff = corr_coeffs[coeff_id]
        if curr_coeff > 0.0:
            diet_positive_names.append(top_features_merged[coeff_id])
            diet_positive_imp.append(top_features_intersection_imp[coeff_id])
        elif curr_coeff < 0.0:
            diet_negative_names.append(top_features_merged[coeff_id])
            diet_negative_imp.append(top_features_intersection_imp[coeff_id])

    diet_positive_imp, diet_positive_names = map(list,
                                                 zip(*sorted(zip(diet_positive_imp, diet_positive_names),
                                                             reverse=False)))
    colors_positive = ['lightslategray', ] * len(diet_positive_names)
    for name_id in range(0, len(diet_positive_names)):
        otu_name_list = diet_positive_names[name_id].split('_')
        otu_name = otu_name_list[0] + '_' + otu_name_list[1]
        if otu_name in top_features_common_with_paper:
            colors_positive[name_id] = 'crimson'

    diet_negative_imp, diet_negative_names = map(list,
                                                 zip(*sorted(zip(diet_negative_imp, diet_negative_names),
                                                             reverse=False)))
    colors_negative = ['lightslategray', ] * len(diet_negative_names)

    for name_id in range(0, len(diet_negative_names)):
        otu_name_list = diet_negative_names[name_id].split('_')
        otu_name = otu_name_list[0] + '_' + otu_name_list[1]
        if otu_name in top_features_common_with_paper:
            colors_negative[name_id] = 'crimson'

    plot_hist(diet_positive_imp, diet_positive_names, colors_positive, 'positive', config.path_out)
    plot_hist(diet_negative_imp, diet_negative_names, colors_negative, 'negative', config.path_out)

    diet_positive_otus_subject = []
    diet_positive_otus_control = []
    diet_negative_otus_subject = []
    diet_negative_otus_control = []

    for otu_id in range(0, len(top_features_merged)):
        otu_gain_count_subject = 0
        otu_loss_count_subject = 0
        otu_gain_count_control = 0
        otu_loss_count_control = 0
        for person_id in range(0, len(common_subjects)):
            person_data_index = config.subject_info_T0['CODE'].index(common_subjects[person_id])
            if config.subject_info_T0['status'][person_data_index] == 'Subject':
                curr_t0 = new_df_t0.iat[person_id, otu_id]
                curr_t1 = new_df_t1.iat[person_id, otu_id]
                if curr_t1 > curr_t0:
                    otu_gain_count_subject += 1
                elif curr_t0 > curr_t1:
                    otu_loss_count_subject += 1
            if config.subject_info_T0['status'][person_data_index] == 'Control':
                curr_t0 = new_df_t0.iat[person_id, otu_id]
                curr_t1 = new_df_t1.iat[person_id, otu_id]
                if curr_t1 > curr_t0:
                    otu_gain_count_control += 1
                elif curr_t0 > curr_t1:
                    otu_loss_count_control += 1

        if top_features_merged[otu_id] in diet_positive_names:
            diet_positive_otus_subject.append(otu_gain_count_subject / otu_loss_count_subject)
            diet_positive_otus_control.append(otu_gain_count_control / otu_loss_count_control)

        if top_features_merged[otu_id] in diet_negative_names:
            diet_negative_otus_subject.append(otu_gain_count_subject / otu_loss_count_subject)
            diet_negative_otus_control.append(otu_gain_count_control / otu_loss_count_control)

    diet_positive_otus = [np.log(diet_positive_otus_subject[i] / diet_positive_otus_control[i]) for i in
                          range(0, len(diet_positive_otus_subject))]
    diet_negative_otus = [np.log(diet_negative_otus_subject[i] / diet_negative_otus_control[i]) for i in
                          range(0, len(diet_negative_otus_subject))]

    plot_box(diet_positive_otus, 'DietPositive', diet_negative_otus, 'DietNegative', config.path_out)


def run_regressor(config, otu_df, adherence_dict, adherence_key, timeline):
    clf = RandomForestRegressor(n_estimators=500, min_samples_split=100)

    output = cross_validate(clf, otu_df, adherence_dict[adherence_key], cv=2,
                            scoring='neg_mean_absolute_error',
                            return_estimator=True)
    output_pred = cross_val_predict(clf, otu_df, adherence_dict[adherence_key], cv=2)
    accuracy = np.mean(output['test_score'])
    is_equal_range = False
    plot_random_forest(adherence_dict[adherence_key], output_pred, timeline, is_equal_range, config.path_out)
    is_equal_range = True
    plot_random_forest(adherence_dict[adherence_key], output_pred, timeline, is_equal_range, config.path_out)

    features_dict = dict((key, []) for key in list(config.otu_col_dict.keys()))
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           index=list(config.otu_col_dict.keys()),
                                           columns=['importance']).sort_values('importance', ascending=False)

        features_names = list(feature_importances.index.values)
        features_values = list(feature_importances.values)
        for feature_id in range(0, len(features_names)):
            features_dict[features_names[feature_id]].append(features_values[feature_id][0])

    for key in features_dict.keys():
        features_dict[key] = np.mean(features_dict[key])
    features_dict = {k: v for k, v in sorted(features_dict.items(), reverse=True, key=lambda x: x[1])}

    accuracy_list = []
    num_features_list = []
    for experiment_id in range(1, 101):
        if experiment_id % 10 == 0:
            print(timeline + ' experiment #', str(experiment_id))
        features_list_len = 5 * experiment_id
        features_list = list(features_dict.keys())[0:features_list_len]
        new_df = otu_df[features_list].copy()
        clf = RandomForestRegressor(n_estimators=500, min_samples_split=100)

        output = cross_validate(clf, new_df, adherence_dict[adherence_key], cv=2,
                                scoring='neg_mean_absolute_error',
                                return_estimator=True)
        accuracy = np.mean(output['test_score'])
        accuracy_list.append(accuracy)
        num_features_list.append(features_list_len)
    plot_scatter(num_features_list, accuracy_list, timeline, config.path_out)

    num_features = 0
    top_features = []
    top_features_imp = []
    for key in features_dict.keys():
        if num_features < 75:
            top_features.append(key)
            top_features_imp.append(features_dict[key])
            num_features += 1

    f = open(config.path_in + '/' + timeline + '_otus.txt', 'w')
    f.write('MAE: ' + str(accuracy) + '\n')
    for item in top_features:
        f.write(item + '\n')
    f.close()

    return top_features, top_features_imp
