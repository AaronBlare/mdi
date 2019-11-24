from config.config import Config
import plotly
import plotly.graph_objects as go
import colorlover as cl
from routines.plot import get_axis, get_margin
from infrastructure.file_system import get_path
import os
import numpy as np
from scipy import stats

box_points = 'outliers'

path = get_path()

in_path = path
out_path = path + '/statistics/cumulated_abundance/supp_fig_10'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

config = Config(in_path, out_path)

common_subjects = config.get_common_subjects_with_adherence()

target_keys = ['compliance160', 'frailty_status']

metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')
metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, target_keys, 'T1')

adh_t0 = []
adh_t1 = []
subjects_frailty_t0 = {}
subjects_frailty_t1 = {}
for code in common_subjects:
    adh_t0.append(metadata_t0[code]['compliance160'])
    adh_t1.append(metadata_t1[code]['compliance160'])

    if metadata_t0[code]['frailty_status'] not in subjects_frailty_t0:
        subjects_frailty_t0[metadata_t0[code]['frailty_status']] = [code]
    else:
        subjects_frailty_t0[metadata_t0[code]['frailty_status']].append(code)

    if metadata_t1[code]['frailty_status'] not in subjects_frailty_t1:
        subjects_frailty_t1[metadata_t1[code]['frailty_status']] = [code]
    else:
        subjects_frailty_t1[metadata_t1[code]['frailty_status']].append(code)

subjects_frailty_t0['Frail'] = []

adh_entire = adh_t0 + adh_t1
common_subjects_entire = common_subjects + common_subjects

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()
common_otu_entire = np.concatenate((common_otu_t0, common_otu_t1), axis=0)


frailty_statuses = ['NonFrail', 'PreFrail', 'Frail']

# plot bar plots
x = ['Baseline', 'Final']
traces = []
for status in frailty_statuses:
    if status in obs_dict_t0['frailty_status']:
        y_t0 = len(obs_dict_t0['frailty_status'][status])
    else:
        y_t0 = 0
    ys = [y_t0, len(obs_dict_t1['frailty_status'][status])]

    traces.append(go.Bar(x=x, y=ys, name=status))

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('Number of subjects')
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/frailty_status_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/frailty_status_box_plot.png')
plotly.io.write_image(fig, out_path + '/frailty_status_box_plot.pdf')

#f = open(path + '/rf_regressor/diet_positive.txt')
f = open(path + '/original/diet_positive.txt')
otus_dp = f.read().splitlines()
f.close()
otus_dp_cols = np.array([common_otu_col_dict[x] for x in otus_dp if x in common_otu_col_dict])

#f = open(path + '/rf_regressor/diet_negative.txt')
f = open(path + '/original/diet_negative.txt')
otus_dn = f.read().splitlines()
f.close()
otus_dn_cols = np.array([common_otu_col_dict[x] for x in otus_dn if x in common_otu_col_dict])

traces_t1_dp = []
traces_entire_dp = []
traces_t1_dn = []
traces_entire_dn = []

ys_t1_dp = []
ys_entire_dp = []
ys_t1_dn = []
ys_entire_dn = []

for status_id, status in enumerate(frailty_statuses):

    t0_ids = np.array([subject_row_dict_T0[x] for x in subjects_frailty_t0[status]], dtype=int)
    t1_ids = np.array([subject_row_dict_T1[x] for x in subjects_frailty_t1[status]], dtype=int)
    t1_for_entire_ids = np.array([subject_row_dict_T1[x] + common_otu_t0.shape[0] for x in subjects_frailty_t1[status]])
    entire_ids = np.concatenate((t0_ids, t1_for_entire_ids))

    color = cl.scales['8']['qual']['Set1'][status_id]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
    line_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    ys_t1 = np.sum(common_otu_t1[np.ix_(t1_ids, otus_dp_cols)], axis=1)
    ys_t1_dp.append(ys_t1)
    trace = go.Box(
        x=ys_t1,
        name=status,
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t1_dp.append(trace)

    ys_entire = np.sum(common_otu_entire[np.ix_(entire_ids, otus_dp_cols)], axis=1)
    ys_entire_dp.append(ys_entire)
    trace = go.Box(
        x=ys_entire,
        name=status,
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_entire_dp.append(trace)

    ys_t1 = np.sum(common_otu_t1[np.ix_(t1_ids, otus_dn_cols)], axis=1)
    ys_t1_dn.append(ys_t1)
    trace = go.Box(
        x=ys_t1,
        name=status,
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t1_dn.append(trace)

    ys_entire = np.sum(common_otu_entire[np.ix_(entire_ids, otus_dn_cols)], axis=1)
    ys_entire_dn.append(ys_entire)
    trace = go.Box(
        x=ys_entire,
        name=status,
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_entire_dn.append(trace)

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=False,
    xaxis=get_axis(''),
    yaxis=get_axis('Cumulated Abundance')
)

fig = go.Figure(data=traces_t1_dp, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t1_diet_positive_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_positive_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_positive_box_plot.pdf')

fig = go.Figure(data=traces_entire_dp, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_entire_diet_positive_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_positive_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_positive_box_plot.pdf')

fig = go.Figure(data=traces_t1_dn, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t1_diet_negative_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_negative_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_negative_box_plot.pdf')

fig = go.Figure(data=traces_entire_dn, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_entire_diet_negative_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_negative_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_negative_box_plot.pdf')

kruskal_t1_dp = stats.kruskal(ys_t1_dp[0], ys_t1_dp[1], ys_t1_dp[2])
print(f'kruskal_t1_dp p-value = {kruskal_t1_dp.pvalue : .2e}')

kruskal_entire_dp = stats.kruskal(ys_entire_dp[0], ys_entire_dp[1], ys_entire_dp[2])
print(f'kruskal_entire_dp p-value = {kruskal_entire_dp.pvalue : .2e}')

kruskal_t1_dn = stats.kruskal(ys_t1_dn[0], ys_t1_dn[1], ys_t1_dn[2])
print(f'kruskal_t1_dn p-value = {kruskal_t1_dn.pvalue : .2e}')

kruskal_entire_dn = stats.kruskal(ys_entire_dn[0], ys_entire_dn[1], ys_entire_dn[2])
print(f'kruskal_entire_dn p-value = {kruskal_entire_dn.pvalue : .2e}')
