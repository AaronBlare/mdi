from rf.plot import plot_box, plot_hist, plot_scatter, plot_heatmap, plot_random_forest
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

    otu_t0_df = pd.DataFrame(otu_t0, common_subjects, list(config.common_otu_col_dict.keys()))
    otu_t1_df = pd.DataFrame(otu_t1, common_subjects, list(config.common_otu_col_dict.keys()))

    top_features_t0, top_features_imp_t0 = run_regressor(config, otu_t0_df, adherence_dict, adherence_key_t0, 'T0')
    top_features_t1, top_features_imp_t1 = run_regressor(config, otu_t1_df, adherence_dict, adherence_key_t1, 'T1')

    top_features_merged = list(set(top_features_t0 + top_features_t1))
    f = open(config.path_out + '/top_OTUs_list.txt', 'w')
    for item in top_features_merged:
        f.write(item + '\n')
    f.close()

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

    otu_file = config.path_in + '/original/' + 'random_forest.txt'
    f = open(otu_file)
    top_features_paper = f.read().splitlines()
    f.close()

    top_features_common_with_paper = list(set(top_features_merged).intersection(set(top_features_paper)))
    print('Number of common OTUs: ', str(len(top_features_common_with_paper)))

    new_df_t0 = otu_t0_df[top_features_merged].copy()
    new_df_t1 = otu_t1_df[top_features_merged].copy()
    new_df = new_df_t0.append(new_df_t1)
    new_adherence = adherence_dict[adherence_key_t0] + adherence_dict[adherence_key_t1]

    metrics_dict = {'otu': [], 'rho': [], 'p_value': [], 'p_value_fdr': []}
    corr_coeffs = []

    for i in range(0, len(top_features_merged)):
        corr_coeff, p_val = spearmanr(list(new_df.iloc[:, i]), new_adherence)
        metrics_dict['otu'].append(top_features_merged[i])
        metrics_dict['rho'].append(corr_coeff)
        metrics_dict['p_value'].append(p_val)
        corr_coeffs.append(corr_coeff)

    reject, pvals_corr, alphacSidak, alphacBonf = multipletests(metrics_dict['p_value'], 0.05, method='fdr_bh')
    metrics_dict['p_value_fdr'] = pvals_corr

    path = config.path_out
    fn_xlsx = path + 'top_OTU_spearman.xlsx'
    df = pd.DataFrame(metrics_dict)
    writer = pd.ExcelWriter(fn_xlsx, engine='xlsxwriter')
    writer.book.use_zip64()
    df.to_excel(writer, index=False)
    writer.save()

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

    top_features_merged_bact = []
    for name_id in range(0, len(top_features_merged)):
        name = top_features_merged[name_id]
        index = name_list.index(name)
        top_features_merged_bact.append(name + '_' + bact_list[index])

    diet_positive_names = []
    diet_positive_names_bact = []
    diet_positive_imp = []
    diet_negative_names = []
    diet_negative_names_bact = []
    diet_negative_imp = []

    for otu_id in range(0, len(metrics_dict['otu'])):
        otu_name = metrics_dict['otu'][otu_id]

        if metrics_dict['p_value_fdr'][otu_id] < 0.05 and metrics_dict['rho'][otu_id] > 0.0:
            diet_positive_names.append(otu_name)
            diet_positive_names_bact.append(top_features_merged_bact[top_features_merged.index(otu_name)])
            diet_positive_imp.append(top_features_intersection_imp[top_features_merged.index(otu_name)])

        elif metrics_dict['p_value_fdr'][otu_id] < 0.05 and metrics_dict['rho'][otu_id] < 0.0:
            diet_negative_names.append(otu_name)
            diet_negative_names_bact.append(top_features_merged_bact[top_features_merged.index(otu_name)])
            diet_negative_imp.append(top_features_intersection_imp[top_features_merged.index(otu_name)])

    f = open(config.path_out + '/diet_positive.txt', 'w')
    for item in diet_positive_names:
        f.write(item + '\n')
    f.close()

    f = open(config.path_out + '/diet_negative.txt', 'w')
    for item in diet_negative_names:
        f.write(item + '\n')
    f.close()

    diet_positive_imp, diet_positive_names_bact = map(list,
                                                 zip(*sorted(zip(diet_positive_imp, diet_positive_names_bact),
                                                             reverse=False)))

    f = open(config.path_in + '/original/' + 'diet_positive.txt')
    diet_positive_article = f.read().splitlines()
    f.close()

    positive_features_common_with_paper = list(set(diet_positive_names).intersection(set(diet_positive_article)))
    print('Number of common positive OTUs: ', str(len(positive_features_common_with_paper)))

    f = open(config.path_in + '/original/' + 'diet_negative.txt')
    diet_negative_article = f.read().splitlines()
    f.close()

    negative_features_common_with_paper = list(set(diet_negative_names).intersection(set(diet_negative_article)))
    print('Number of common negative OTUs: ', str(len(negative_features_common_with_paper)))

    colors_positive = ['lightslategray', ] * len(diet_positive_names_bact)
    for name_id in range(0, len(diet_positive_names_bact)):
        otu_name_list = diet_positive_names_bact[name_id].split('_')
        otu_name = otu_name_list[0] + '_' + otu_name_list[1]
        if otu_name in diet_positive_article:
            colors_positive[name_id] = 'crimson'
        if otu_name in diet_negative_article:
            print(otu_name + ' is positive, but in article - negative')

    diet_negative_imp, diet_negative_names_bact = map(list,
                                                 zip(*sorted(zip(diet_negative_imp, diet_negative_names_bact),
                                                             reverse=False)))
    colors_negative = ['lightslategray', ] * len(diet_negative_names_bact)

    for name_id in range(0, len(diet_negative_names_bact)):
        otu_name_list = diet_negative_names_bact[name_id].split('_')
        otu_name = otu_name_list[0] + '_' + otu_name_list[1]
        if otu_name in diet_negative_article:
            colors_negative[name_id] = 'crimson'
        if otu_name in diet_positive_names:
            print(otu_name + ' is negative, but in article - positive')

    plot_hist(diet_positive_imp, diet_positive_names_bact, colors_positive, 'positive', config.path_out)
    plot_hist(diet_negative_imp, diet_negative_names_bact, colors_negative, 'negative', config.path_out)

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

    plot_box([diet_positive_otus, diet_negative_otus], ['DietPositive', 'DietNegative'], config.path_out, 'diet_otus')


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

    features_dict = dict((key, []) for key in list(config.common_otu_col_dict.keys()))
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           index=list(config.common_otu_col_dict.keys()),
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
    for experiment_id in range(1, 1):
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
    plot_scatter(num_features_list, accuracy_list, 'neg_MAE', timeline, config.path_out)

    num_features = 0
    top_features = []
    top_features_imp = []
    for key in features_dict.keys():
        if num_features < 75:
            top_features.append(key)
            top_features_imp.append(features_dict[key])
            num_features += 1

    f = open(config.path_out + '/' + timeline + '_otus.txt', 'w')
    f.write('MAE: ' + str(accuracy) + '\n')
    for item in top_features:
        f.write(item + '\n')
    f.close()

    return top_features, top_features_imp


def run_seq_regressor(otu_df, subject_key):
    clf = RandomForestRegressor(n_estimators=500)
    output = cross_validate(clf, otu_df, subject_key, cv=2, return_estimator=True)
    features_dict = dict((key, []) for key in list(otu_df.columns.values))
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           index=list(otu_df.columns.values),
                                           columns=['importance']).sort_values('importance', ascending=False)
        features_names = list(feature_importances.index.values)
        features_values = list(feature_importances.values)
        for feature_id in range(0, len(features_names)):
            features_dict[features_names[feature_id]].append(features_values[feature_id][0])

    for key in features_dict.keys():
        features_dict[key] = np.mean(features_dict[key])
    features_dict = {k: v for k, v in sorted(features_dict.items(), reverse=True, key=lambda x: x[1])}
    top_features = list(features_dict.keys())

    correlation_list = []
    mae_list = []
    num_features_list = []
    for experiment_id in range(1, 201):
        if experiment_id % 10 == 0:
            print('Experiment #', str(experiment_id))
        features_list_len = experiment_id
        features_list = list(features_dict.keys())[0:features_list_len]
        new_df = otu_df[features_list].copy()
        clf = RandomForestRegressor(n_estimators=500)
        output_pred = cross_val_predict(clf, new_df, subject_key, cv=2)
        correlation_list.append(abs(spearmanr(subject_key, output_pred)[0]))
        mae_list.append(mean_absolute_error(subject_key, output_pred))
        num_features_list.append(features_list_len)

    return top_features, correlation_list, mae_list, num_features_list


def run_regressor_mae_mse(otu_df, subject_key):
    clf = RandomForestRegressor(n_estimators=500)
    output_pred = cross_val_predict(clf, otu_df, subject_key, cv=2)
    mse_list = []
    rmse_list = []
    mae_list = []
    for i in range(0, len(subject_key)):
        curr_mse = mean_squared_error([subject_key[i]], [output_pred[i]], squared=True)
        curr_rmse = mean_squared_error([subject_key[i]], [output_pred[i]], squared=False)
        curr_mae = mean_absolute_error([subject_key[i]], [output_pred[i]])

        mse_list.append(curr_mse)
        rmse_list.append(curr_rmse)
        mae_list.append(curr_mae)
    return mse_list, rmse_list, mae_list


def save_list(config, data, suffix):
    f = open(config.path_out + '/' + suffix + '_otus.txt', 'w')
    for item in data:
        f.write(item + '\n')
    f.close()


def pipeline_regressor_countries(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otus = config.get_common_otus()

    common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

    countries = ['Italy', 'UK', 'Holland', 'Poland', 'France']

    adherence_key = 'compliance160'
    age_key = 'age'
    country_key = 'country'

    target_keys = [adherence_key, age_key, country_key]

    common_subjects = config.get_common_subjects_with_adherence()
    metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')
    metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, target_keys, 'T1')

    subjects_country = {}
    for subject in common_subjects:
        country = metadata_t0[subject][country_key]
        if country in subjects_country:
            subjects_country[country].append(subject)
        else:
            subjects_country[country] = [subject]

    otu_country_data = {}
    for country in countries:
        otu_country_data[country] = np.zeros((len(subjects_country[country]) * 2, len(common_otus)), dtype=np.float32)

    subjects_names = {key: [] for key in countries}
    adherence = {key: [] for key in countries}
    age = {key: [] for key in countries}

    for country in countries:
        for sub_id, sub in enumerate(subjects_country[country]):
            curr_adherence_t0 = metadata_t0[sub][adherence_key]
            curr_adherence_t1 = metadata_t1[sub][adherence_key]

            adherence[country].append(curr_adherence_t0)
            adherence[country].append(curr_adherence_t1)

            curr_age_t0 = metadata_t0[sub][age_key]
            curr_age_t1 = metadata_t1[sub][age_key]

            age[country].append(curr_age_t0)
            age[country].append(curr_age_t1)

            curr_otu_t0 = common_otu_t0[subject_row_dict_T0[sub], :]
            curr_otu_t1 = common_otu_t1[subject_row_dict_T1[sub], :]

            subjects_names[country].append(sub + '_T0')
            subjects_names[country].append(sub + '_T1')

            otu_country_data[country][sub_id * 2, :] = curr_otu_t0
            otu_country_data[country][sub_id * 2 + 1, :] = curr_otu_t1

    mse_adherence = {key: [] for key in countries}
    mae_adherence = {key: [] for key in countries}
    rmse_adherence = {key: [] for key in countries}

    mse_age = {key: [] for key in countries}
    rmse_age = {key: [] for key in countries}
    mae_age = {key: [] for key in countries}
    for country in countries:
        otu_df = pd.DataFrame(otu_country_data[country],
                              subjects_names[country],
                              list(config.common_otu_col_dict.keys()))

        curr_mse_adh, curr_rmse_adh, curr_mae_adh = run_regressor_mae_mse(otu_df, adherence[country])
        curr_mse_age, curr_rmse_age, curr_mae_age = run_regressor_mae_mse(otu_df, age[country])

        mse_adherence[country] = curr_mse_adh
        mae_adherence[country] = curr_mae_adh
        rmse_adherence[country] = curr_rmse_adh

        mse_age[country] = curr_mse_age
        mae_age[country] = curr_mae_age
        rmse_age[country] = curr_rmse_age

    plot_box(mse_adherence, list(mse_adherence.keys()), config.path_out, 'MSE_adh')
    plot_box(rmse_adherence, list(rmse_adherence.keys()), config.path_out, 'RMSE_adh')
    plot_box(mae_adherence, list(mae_adherence.keys()), config.path_out, 'MAE_adh')

    plot_box(mse_age, list(mse_age.keys()), config.path_out, 'MSE_age')
    plot_box(rmse_age, list(rmse_age.keys()), config.path_out, 'RMSE_age')
    plot_box(mae_age, list(mae_age.keys()), config.path_out, 'MAE_age')


def pipeline_seq_regressor_countries(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otus = config.get_common_otus()

    common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

    countries = ['Italy', 'UK', 'Holland', 'Poland', 'France']

    adherence_key = 'compliance160'
    age_key = 'age'
    country_key = 'country'

    target_keys = [adherence_key, age_key, country_key]

    common_subjects = config.get_common_subjects_with_adherence()
    metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')
    metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, target_keys, 'T1')

    subjects_country = {}
    for subject in common_subjects:
        country = metadata_t0[subject][country_key]
        if country in subjects_country:
            subjects_country[country].append(subject)
        else:
            subjects_country[country] = [subject]

    otu_country_data = {}
    for country in countries:
        otu_country_data[country] = np.zeros((len(subjects_country[country]) * 2, len(common_otus)), dtype=np.float32)

    subjects_names = {key: [] for key in countries}
    adherence = {key: [] for key in countries}
    age = {key: [] for key in countries}

    for country in countries:
        for sub_id, sub in enumerate(subjects_country[country]):
            curr_adherence_t0 = metadata_t0[sub][adherence_key]
            curr_adherence_t1 = metadata_t1[sub][adherence_key]

            adherence[country].append(curr_adherence_t0)
            adherence[country].append(curr_adherence_t1)

            curr_age_t0 = metadata_t0[sub][age_key]
            curr_age_t1 = metadata_t1[sub][age_key]

            age[country].append(curr_age_t0)
            age[country].append(curr_age_t1)

            curr_otu_t0 = common_otu_t0[subject_row_dict_T0[sub], :]
            curr_otu_t1 = common_otu_t1[subject_row_dict_T1[sub], :]

            subjects_names[country].append(sub + '_T0')
            subjects_names[country].append(sub + '_T1')

            otu_country_data[country][sub_id * 2, :] = curr_otu_t0
            otu_country_data[country][sub_id * 2 + 1, :] = curr_otu_t1

    for country in countries:
        otu_df = pd.DataFrame(otu_country_data[country],
                              subjects_names[country],
                              list(config.common_otu_col_dict.keys()))

        top_otu_adh, corr_list, mae_list, num_features = run_seq_regressor(otu_df, adherence[country])
        plot_scatter(num_features, corr_list, 'Correlation coefficient', country + '_adh_corr', config.path_out)
        plot_scatter(num_features, mae_list, 'MAE', country + '_adh_mae', config.path_out)
        save_list(config, top_otu_adh, country + '_adh')

        top_otu_age, corr_list, mae_list, num_features = run_seq_regressor(otu_df, age[country])
        plot_scatter(num_features, corr_list, 'Correlation coefficient', country + '_age_corr', config.path_out)
        plot_scatter(num_features, mae_list, 'MAE', country + '_age_mae', config.path_out)
        save_list(config, top_otu_age, country + '_age')
