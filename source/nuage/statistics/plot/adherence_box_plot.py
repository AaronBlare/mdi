from config.config import Config
import plotly
import plotly.graph_objects as go
import colorlover as cl
from routines.plot import get_axis, get_margin
from infrastructure.file_system import get_path
import os

countries = ['Italy', 'UK', 'Holland', 'Poland', 'France']

path = get_path()

in_path = path
out_path = path + '/statistics/adherence'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

config = Config(in_path, out_path)

common_subjects = config.get_common_subjects_with_adherence()

target_keys = ['country', 'compliance160']

metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, target_keys, 'T0')

traces = []

for country in countries:
    codes = obs_dict_t0['country'][country]

    ys = []
    for code in codes:
        ys.append(float(metadata_t0[code]['compliance160']) / 160.0 * 100.0 )

    color = cl.scales['8']['qual']['Set1'][countries.index(country)]
    coordinates = color[4:-1].split(',')
    marker_color = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
    line_color = 'rgba(' + ','.join(coordinates) + ',' + str(1.0) + ')'

    trace = go.Box(
        y=ys,
        name=country,
        boxpoints='outliers',
        marker_color=marker_color,
        line_color=line_color
    )

    traces.append(trace)

layout = go.Layout(
    margin=get_margin(),
    autosize=True,
    showlegend=False,
    xaxis=get_axis(''),
    yaxis=get_axis('adherence')
)

fig = go.Figure(data=traces, layout=layout)

plotly.offline.plot(fig, filename=out_path + '/adherence_t0_countries_box_plot.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + '/adherence_t0_countries_box_plot.png')
plotly.io.write_image(fig, out_path + '/adherence_t0_countries_box_plot.pdf')
