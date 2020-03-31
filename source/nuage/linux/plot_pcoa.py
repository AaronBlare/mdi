import plotly
import colorlover as cl
import plotly.graph_objs as go
import os.path
from routines.plot import get_axis, get_margin, get_legend


def pcoa_plot(path, ord_result, common_subjects):
    coord_matrix = ord_result.samples.values.T
    xs_all = coord_matrix[0]
    ys_all = coord_matrix[1]
    zs_all = coord_matrix[2]

    traces_3d = []
    traces_2d = []

    xs = []
    ys = []
    zs = []
    for subj in common_subjects:
        index = common_subjects.index(subj)
        xs.append(xs_all[index])
        ys.append(ys_all[index])
        zs.append(zs_all[index])

    color = cl.scales['8']['qual']['Set1'][0]
    coordinates = color[4:-1].split(',')
    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.3) + ')'
    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        name='Italy',
        mode='markers',
        marker=dict(
            size=8,
            color=color_border,
            line=dict(
                color=color_transparent,
                width=0.5
            ),
            opacity=0.8
        )
    )
    traces_3d.append(trace)

    trace = go.Scatter(
        x=ys,
        y=xs,
        name='Italy',
        mode='markers',
        marker=dict(
            size=8,
            color=color_border,
            line=dict(
                color=color_transparent,
                width=0.5
            ),
            opacity=0.8
        )
    )
    traces_2d.append(trace)

    layout_3d = go.Layout(
        margin=get_margin(),
        autosize=True,
        legend=get_legend()
    )

    layout_2d = go.Layout(
        margin=get_margin(),
        autosize=True,
        legend=get_legend(),
        xaxis=get_axis("PC1"),
        yaxis=get_axis("PC2")
    )

    fig_3d = go.Figure(data=traces_3d, layout=layout_3d)
    fig_3d.update_layout(scene=dict(
        xaxis=get_axis("PC1"),
        yaxis=get_axis("PC2"),
        zaxis=get_axis("PC3")
    ))
    fig_2d = go.Figure(data=traces_2d, layout=layout_2d)

    if not os.path.exists(path):
        os.makedirs(path)

    plotly.offline.plot(fig_3d, filename=path + '/pcoa_3d.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig_3d, path + '/pcoa_3d.png')
    plotly.io.write_image(fig_3d, path + '/pcoa_3d.pdf')

    plotly.offline.plot(fig_2d, filename=path + '/pcoa_2d.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig_2d, path + '/pcoa_2d.png')
    plotly.io.write_image(fig_2d, path + '/pcoa_2d.pdf')
