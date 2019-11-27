import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly
import numpy as np

def cmocean_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        tmp = np.array(cmap(k*h)[:3])*255
        C = []
        for i in tmp:
            C.append(np.uint8(i))
        C = np.array(C)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def get_axis(title):
    axis = dict(
        title=title,
        showgrid=True,
        showline=True,
        mirror='ticks',
        titlefont=dict(
            family='Arial',
            color='black',
            size=24,
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            family='Arial',
            color='black',
            size=20
        ),
        exponentformat='e',
        showexponent='all'
    )
    return axis


def get_margin():
    margin = go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    )
    return margin


def get_legend():
    legend = dict(
        font=dict(
            family='Arial',
            size=16,
        )
    )
    return legend


def plot_table_pdf(path_in, path_out, fn, header):
    fn_load = path_in + fn + '.xlsx'
    table_dict = load_table_dict_xlsx(fn_load)
    hist_data = [table_dict[header]]
    group_labels = [header]

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False,  curve_type='normal')

    layout = go.Layout(
        showlegend=False,
        margin=get_margin(),
        autosize=True,
        legend=get_legend(),
        xaxis=get_axis(header),
        yaxis=get_axis('PDF')
    )
    fig['layout'] = layout

    fn_save = path_out + fn + '_' + header + '_pdf'
    plotly.io.write_image(fig, fn_save + '.png')
    plotly.io.write_image(fig, fn_save + '.pdf')
    plotly.offline.plot(fig, filename=fn_save + '.html', auto_open=False, show_link=True)