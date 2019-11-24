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

markers_original_mse = run_regressor(markers_original_df, adherence[countries.index(target_country)])
markers_original_rmse = np.sqrt(markers_original_mse)

non_markers_original_mse = run_regressor(non_markers_original_df, adherence[countries.index(target_country)])
non_markers_original_rmse = np.sqrt(non_markers_original_mse)

markers_rf_mse = run_regressor(markers_rf_df, adherence[countries.index(target_country)])
markers_rf_rmse = np.sqrt(markers_rf_mse)

non_markers_rf_mse = run_regressor(non_markers_rf_df, adherence[countries.index(target_country)])
non_markers_rf_rmse = np.sqrt(non_markers_rf_mse)

traces = []
traces.append(go.Box(
    y=markers_original_rmse,
    name='Markers',
    boxpoints=box_points
))
traces.append(go.Box(
    y=non_markers_original_rmse,
    name='Non-Markers',
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
