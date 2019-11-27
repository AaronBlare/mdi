from config.config import Config
from infrastructure.file_system import get_path
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from routines.plot import get_axis, get_margin
import plotly
import plotly.graph_objects as go
import colorlover as cl



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

adherence = {}
subjects_country = {}
for country in countries:
    codes = obs_dict['country'][country]

    for code in codes:

        if country in adherence:
            adherence[country].append(metadata[code]['compliance160'])
        else:
            adherence[country] = [metadata[code]['compliance160']]

        if metadata[code]['country'] not in subjects_country:
            subjects_country[metadata[code]['country']] = [code]
        else:
            subjects_country[metadata[code]['country']].append(code)

common_otus = config.get_common_otus()
common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

otu_data = {}
for country in countries:
    otu_data[country] = np.zeros((len(subjects_country[country]), len(common_otus)), dtype=np.float32)

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0

for country in countries:
    for sub_id, sub in enumerate(subjects_country[country]):
        curr_otu = common_otu_t0[subject_row_dict_T0[sub], :]
        otu_data[country][sub_id, :] = curr_otu

otu_df = {}
for country in countries:
    otu_df[country] = pd.DataFrame(otu_data[country], subjects_country[country],
                                        list(common_otu_col_dict.keys()))

mse = {}
rmse = {}
for country in countries:
    mse[country] = run_regressor(otu_df[country], adherence[country])
    rmse[country] = np.sqrt(mse[country])

traces = []
for country in countries:
    ys = rmse[country]

    color = cl.scales['8']['qual']['Set1'][countries.index(country)]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
    line_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

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
