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

dr_otus_source = 'original'

path = get_path()

in_path = path
out_path = path + '/statistics/frailty_changes/supp_fig_10_c'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

config = Config(in_path, out_path)

common_subjects = config.get_common_subjects_with_adherence()

target_keys = ['compliance160', 'frailty_status', 'status']

metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')
metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, target_keys, 'T1')

def fs_to_int(fs):
    if fs == 'NonFrail':
        return 0
    elif fs == 'PreFrail':
        return 1
    else:
        return 2

frailty_changes = {}
for code in common_subjects:
    fc = fs_to_int(metadata_t1[code]['frailty_status']) - fs_to_int(metadata_t0[code]['frailty_status'])
    frailty_changes[code] = fc

frailty_changes_count = {}
for status, codes in obs_dict_t0['status'].items():
    frailty_changes_count[status] = {
        'Reduced Frailty': 0,
        'No change in Frailty': 0,
        'Increased Frailty': 0
    }
    for code in codes:
        if frailty_changes[code] < 0:
            frailty_changes_count[status]['Reduced Frailty'] += 1
        elif frailty_changes[code] == 0:
            frailty_changes_count[status]['No change in Frailty'] += 1
        else:
            frailty_changes_count[status]['Increased Frailty'] += 1

traces = []
xs = ['Intervention', 'Control']
for color_id, fc_type in  enumerate(['Reduced Frailty', 'No change in Frailty', 'Increased Frailty']):

    color = cl.scales['8']['qual']['Set1'][color_id]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    ys =  [
        frailty_changes_count['Subject'][fc_type],
        frailty_changes_count['Control'][fc_type],
    ]
    traces.append(
        go.Bar(
            x=xs,
            y=ys,
            name=fc_type,
            marker_color=marker_color
        )
    )

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('Number of subjects')
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/frailty_changes_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/frailty_changes_box_plot.png')
plotly.io.write_image(fig, out_path + '/frailty_changes_box_plot.pdf')


traces = []
xs = ['Intervention', 'Control']
for fc_type in ['Reduced Frailty', 'Increased Frailty']:

    if fc_type == 'Reduced Frailty':
        color_id = 0
    else:
        color_id = 2

    color = cl.scales['8']['qual']['Set1'][color_id]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    ys =  [
        float(frailty_changes_count['Subject'][fc_type]) / float(frailty_changes_count['Subject']['Reduced Frailty'] + frailty_changes_count['Subject']['Increased Frailty']) * 100,
        float(frailty_changes_count['Control'][fc_type]) / float(frailty_changes_count['Control']['Reduced Frailty'] + frailty_changes_count['Control']['Increased Frailty']) * 100,
    ]
    traces.append(
        go.Bar(
            x=xs,
            y=ys,
            name=fc_type,
            marker_color=marker_color
        )
    )

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=True,
    xaxis=get_axis(''),
    yaxis=get_axis('Percent of subjects'),
    barmode='stack'
)

fig = go.Figure(data=traces, layout=layout)
plotly.offline.plot(fig, filename=out_path + '/frailty_inc_dec_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/frailty_inc_dec_box_plot.png')
plotly.io.write_image(fig, out_path + '/frailty_inc_dec_box_plot.pdf')