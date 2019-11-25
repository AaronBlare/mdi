from config.config import Config
from infrastructure.file_system import get_path
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import mean_squared_error
from routines.plot import get_axis, get_margin
import plotly
import plotly.graph_objects as go


def run_regressor(otu_df, adherence):
    clf = RandomForestRegressor(n_estimators=500)
    output_pred = cross_val_predict(clf, otu_df, adherence, cv=2)
    mse_list = [mean_squared_error([adherence[i]], [output_pred[i]]) for i in range(0, len(adherence))]
    return mse_list


def run_iterative_regressor(otu_df, adherence, otu_list, suffix):
    clf = RandomForestRegressor(n_estimators=500)
    output = cross_validate(clf, otu_df, adherence, cv=2, return_estimator=True)
    features_dict = dict((key, []) for key in otu_list)
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           index=otu_list,
                                           columns=['importance']).sort_values('importance', ascending=False)

        features_names = list(feature_importances.index.values)
        features_values = list(feature_importances.values)
        for feature_id in range(0, len(features_names)):
            features_dict[features_names[feature_id]].append(features_values[feature_id][0])

    for key in features_dict.keys():
        features_dict[key] = np.mean(features_dict[key])
    features_dict = {k: v for k, v in sorted(features_dict.items(), reverse=True, key=lambda x: x[1])}

    mse_list = []
    for experiment_id in range(1, 101):
        if experiment_id % 10 == 0:
            print(suffix + ' experiment #', str(experiment_id))
        features_list = list(features_dict.keys())[0:experiment_id]
        new_df = otu_df[features_list].copy()
        clf = RandomForestRegressor(n_estimators=500)
        output_pred = cross_val_predict(clf, new_df, adherence, cv=2)
        mse_list.append(mean_squared_error(adherence, output_pred))
    return mse_list


box_points = 'outliers'

path = get_path()

in_path = path
out_path = path + '/rf_regressor/supp_fig_7'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

config = Config(in_path, out_path)

common_subjects = config.get_common_subjects_with_adherence()

target_keys = ['compliance160', 'country']
countries = ['Italy', 'UK', 'Holland', 'Poland', 'France']

metadata, obs_dict = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')

adherence = [[] for i in range(0, len(countries))]
subjects_country = {}
for i, country in enumerate(countries):
    codes = obs_dict['country'][country]

    for code in codes:
        adherence[i].append(metadata[code]['compliance160'])

        if metadata[code]['country'] not in subjects_country:
            subjects_country[metadata[code]['country']] = [code]
        else:
            subjects_country[metadata[code]['country']].append(code)

common_otus = config.get_common_otus()
common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

otu_data = {}
for i in range(0, len(countries)):
    otu_data[countries[i]] = np.zeros((len(subjects_country[countries[i]]), len(common_otus)), dtype=np.float32)

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0

for i in range(0, len(countries)):
    for sub_id, sub in enumerate(subjects_country[countries[i]]):
        curr_otu = common_otu_t0[subject_row_dict_T0[sub], :]
        otu_data[countries[i]][sub_id, :] = curr_otu

otu_df = {}
for i in range(0, len(countries)):
    otu_df[countries[i]] = pd.DataFrame(otu_data[countries[i]], subjects_country[countries[i]],
                                        list(common_otu_col_dict.keys()))

mse = {}
rmse = {}
for i in range(0, len(countries)):
    mse[countries[i]] = run_regressor(otu_df[countries[i]], adherence[i])
    rmse[countries[i]] = np.sqrt(mse[countries[i]])

traces = []
for country in countries:
    ys = rmse[country]
    traces.append(go.Box(
        y=ys,
        name=country,
        boxpoints=box_points
    ))

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('RMSE across countries at T0')
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/country_t0_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/country_t0_box_plot.png')
plotly.io.write_image(fig, out_path + '/country_t0_box_plot.pdf')

target_country = 'Holland'

markers_original_fn = path + '/original/random_forest.txt'
f = open(markers_original_fn)
markers_original = f.read().splitlines()
f.close()
markers_original = list(set(common_otu_col_dict.keys()).intersection(set(markers_original)))
non_markers_original = list(set(common_otu_col_dict.keys()).difference(set(markers_original)))

markers_rf_fn = path + '/rf_regressor/top_OTUs_list.txt'
f = open(markers_rf_fn)
markers_rf = f.read().splitlines()
f.close()
markers_rf = list(set(common_otu_col_dict.keys()).intersection(set(markers_rf)))
non_markers_rf = list(set(common_otu_col_dict.keys()).difference(set(markers_rf)))

markers_original_df = otu_df[target_country][markers_original]
non_markers_original_df = otu_df[target_country][non_markers_original]
markers_rf_df = otu_df[target_country][markers_rf]
non_markers_rf_df = otu_df[target_country][non_markers_rf]

markers_original_mse = run_iterative_regressor(markers_original_df, adherence[countries.index(target_country)],
                                               markers_original, 'Markers original')
non_markers_original_mse = run_iterative_regressor(non_markers_original_df, adherence[countries.index(target_country)],
                                                   non_markers_original, 'Non-Markers original')

traces = []
traces.append(go.Box(
    y=markers_original_mse,
    name='Markers',
    boxpoints=box_points
))
traces.append(go.Box(
    y=non_markers_original_mse,
    name='Non-Markers',
    boxpoints=box_points
))

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('MSE across iterative models at Baseline')
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/holland_original_t0_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/holland_original_t0_box_plot.png')
plotly.io.write_image(fig, out_path + '/holland_original_t0_box_plot.pdf')

markers_rf_mse = run_iterative_regressor(markers_rf_df, adherence[countries.index(target_country)],
                                         markers_rf, 'Markers rf')
non_markers_rf_mse = run_iterative_regressor(non_markers_rf_df, adherence[countries.index(target_country)],
                                             non_markers_rf, 'Non-Markers rf')

traces = []
traces.append(go.Box(
    y=markers_rf_mse,
    name='Markers',
    boxpoints=box_points
))
traces.append(go.Box(
    y=non_markers_rf_mse,
    name='Non-Markers',
    boxpoints=box_points
))

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('MSE across iterative models at Baseline')
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/holland_rf_t0_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/holland_rf_t0_box_plot.png')
plotly.io.write_image(fig, out_path + '/holland_rf_t0_box_plot.pdf')
