from config.config import Config
import plotly
import plotly.graph_objects as go
import colorlover as cl
from routines.plot import get_axis, get_margin
from infrastructure.file_system import get_path
import os
import numpy as np
from scipy import stats


num_segments = 6
num_windows = 5

box_points = 'outliers'

path = get_path()

in_path = path
out_path = path + '/statistics/cumulated_abundance/supp_fig_8'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

config = Config(in_path, out_path)

common_subjects = config.get_common_subjects_with_adherence()

target_keys = ['compliance160']

metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')
metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, target_keys, 'T1')

adh_t0 = []
adh_t1 = []
for code in common_subjects:
    adh_t0.append(metadata_t0[code]['compliance160'])
    adh_t1.append(metadata_t1[code]['compliance160'])
adh_entire = adh_t0 + adh_t1
common_subjects_entire = common_subjects + common_subjects

borders_t0 = np.linspace(0, len(adh_t0) - 1, num_segments + 1, dtype=int)
borders_t1 = np.linspace(0, len(adh_t1) - 1, num_segments + 1, dtype=int)
borders_entire = np.linspace(0, len(adh_entire) - 1, num_segments + 1, dtype=int)

order_t0 = np.argsort(adh_t0)
order_t1 = np.argsort(adh_t1)
order_entire = np.argsort(adh_entire)

adh_t0_sorted = np.array(adh_t0)[order_t0]
adh_t1_sorted = np.array(adh_t1)[order_t1]
adh_entire_sorted = np.array(adh_entire)[order_entire]

segments_t0 = {}
segments_t1 = {}
segments_entire = {}
for s_id in range(0, num_segments):

    indexes_t0 = order_t0[borders_t0[s_id] : borders_t0[s_id + 1]]
    indexes_t1 = order_t1[borders_t1[s_id] : borders_t1[s_id + 1]]
    indexes_entire = order_entire[borders_entire[s_id]: borders_entire[s_id + 1]]

    codes_t0 = np.array(common_subjects)[indexes_t0]
    codes_t1 = np.array(common_subjects)[indexes_t1]
    codes_entire = np.array(common_subjects_entire)[indexes_entire]

    segments_t0[s_id] = codes_t0
    segments_t1[s_id] = codes_t1
    segments_entire[s_id] = codes_entire

windows_t0 = {}
windows_t1 = {}
windows_entire = {}
for w_id in range(0, num_windows):
    windows_t0[w_id] = np.concatenate((segments_t0[w_id], segments_t0[w_id + 1]))
    windows_t1[w_id] = np.concatenate((segments_t1[w_id], segments_t1[w_id + 1]))
    windows_entire[w_id] = np.concatenate((segments_entire[w_id], segments_entire[w_id + 1]))

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()
common_otu_entire = np.concatenate((common_otu_t0, common_otu_t1), axis=0)

f = open(path + '/rf_regressor/diet_positive.txt')
otus_dp = f.read().splitlines()
f.close()
otus_dp_cols = np.array([common_otu_col_dict[x] for x in otus_dp if x in common_otu_col_dict])

f = open(path + '/rf_regressor/diet_negative.txt')
otus_dn = f.read().splitlines()
f.close()
otus_dn_cols = np.array([common_otu_col_dict[x] for x in otus_dn if x in common_otu_col_dict])

traces_t0_dp = []
traces_t1_dp = []
traces_entire_dp = []
traces_t0_dn = []
traces_t1_dn = []
traces_entire_dn = []

ys_t0_dp = []
ys_t1_dp = []
ys_entire_dp = []
ys_t0_dn = []
ys_t1_dn = []
ys_entire_dn = []

for w_id in range(0, num_windows):

    windows_t0_ids = np.array([subject_row_dict_T0[x] for x in windows_t0[w_id]])
    windows_t1_ids = np.array([subject_row_dict_T1[x] for x in windows_t1[w_id]])
    windows_t1_for_entire_ids = np.array([subject_row_dict_T1[x] + common_otu_t0.shape[0] for x in windows_t1[w_id]])
    windows_entire_ids = np.concatenate((windows_t0_ids, windows_t1_for_entire_ids))

    color = cl.scales['8']['seq']['Reds'][w_id + 3]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
    line_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    ys_t0 = np.sum(common_otu_t0[np.ix_(windows_t0_ids, otus_dp_cols)], axis=1)
    ys_t0_dp.append(ys_t0)
    trace = go.Box(
        y=ys_t0,
        name="W" + str(w_id + 1),
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t0_dp.append(trace)

    ys_t1 = np.sum(common_otu_t1[np.ix_(windows_t1_ids, otus_dp_cols)], axis=1)
    ys_t1_dp.append(ys_t1)
    trace = go.Box(
        y=ys_t1,
        name="W" + str(w_id + 1),
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t1_dp.append(trace)

    ys_entire = np.sum(common_otu_entire[np.ix_(windows_entire_ids, otus_dp_cols)], axis=1)
    ys_entire_dp.append(ys_entire)
    trace = go.Box(
        y=ys_entire,
        name="W" + str(w_id + 1),
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_entire_dp.append(trace)

    color = cl.scales['8']['seq']['Blues'][w_id + 3]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
    line_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    ys_t0 = np.sum(common_otu_t0[np.ix_(windows_t0_ids, otus_dn_cols)], axis=1)
    ys_t0_dn.append(ys_t0)
    trace = go.Box(
        y=ys_t0,
        name="W" + str(w_id + 1),
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t0_dn.append(trace)

    ys_t1 = np.sum(common_otu_t1[np.ix_(windows_t1_ids, otus_dn_cols)], axis=1)
    ys_t1_dn.append(ys_t1)
    trace = go.Box(
        y=ys_t1,
        name="W" + str(w_id + 1),
        boxpoints=box_points,
        marker_color=marker_color,
        line_color=line_color
    )
    traces_t1_dn.append(trace)

    ys_entire = np.sum(common_otu_entire[np.ix_(windows_entire_ids, otus_dn_cols)], axis=1)
    ys_entire_dn.append(ys_entire)
    trace = go.Box(
        y=ys_entire,
        name="W" + str(w_id + 1),
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

fig = go.Figure(data=traces_t0_dp, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t0_diet_positive_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t0_diet_positive_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t0_diet_positive_box_plot.pdf')

fig = go.Figure(data=traces_t1_dp, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t1_diet_positive_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_positive_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_positive_box_plot.pdf')

fig = go.Figure(data=traces_entire_dp, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_entire_diet_positive_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_positive_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_positive_box_plot.pdf')

fig = go.Figure(data=traces_t0_dn, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t0_diet_negative_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t0_diet_negative_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t0_diet_negative_box_plot.pdf')

fig = go.Figure(data=traces_t1_dn, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_t1_diet_negative_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_negative_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_t1_diet_negative_box_plot.pdf')

fig = go.Figure(data=traces_entire_dn, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/cumulated_abundance_entire_diet_negative_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_negative_box_plot.png')
plotly.io.write_image(fig, out_path + '/cumulated_abundance_entire_diet_negative_box_plot.pdf')

kruskal_t0_dp = stats.kruskal(ys_t0_dp[0], ys_t0_dp[1], ys_t0_dp[2], ys_t0_dp[3], ys_t0_dp[4])
print(f'kruskal_t0_dp p-value = {kruskal_t0_dp.pvalue : .2e}')

kruskal_t1_dp = stats.kruskal(ys_t1_dp[0], ys_t1_dp[1], ys_t1_dp[2], ys_t1_dp[3], ys_t1_dp[4])
print(f'kruskal_t1_dp p-value = {kruskal_t1_dp.pvalue : .2e}')

kruskal_entire_dp = stats.kruskal(ys_entire_dp[0], ys_entire_dp[1], ys_entire_dp[2], ys_entire_dp[3], ys_entire_dp[4])
print(f'kruskal_entire_dp p-value = {kruskal_entire_dp.pvalue : .2e}')


kruskal_t0_dn = stats.kruskal(ys_t0_dn[0], ys_t0_dn[1], ys_t0_dn[2], ys_t0_dn[3], ys_t0_dn[4])
print(f'kruskal_t0_dn p-value = {kruskal_t0_dn.pvalue : .2e}')

kruskal_t1_dn = stats.kruskal(ys_t1_dn[0], ys_t1_dn[1], ys_t1_dn[2], ys_t1_dn[3], ys_t1_dn[4])
print(f'kruskal_t1_dn p-value = {kruskal_t1_dn.pvalue : .2e}')

kruskal_entire_dn = stats.kruskal(ys_entire_dn[0], ys_entire_dn[1], ys_entire_dn[2], ys_entire_dn[3], ys_entire_dn[4])
print(f'kruskal_entire_dn p-value = {kruskal_entire_dn.pvalue : .2e}')
