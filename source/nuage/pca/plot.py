import plotly
import colorlover as cl
import plotly.graph_objs as go
import os.path

def pca_plot(path, pcs, common_subjects, metrics_key, metrics_dict):

    xs_all = pcs[:, 0]
    ys_all = pcs[:, 1]

    traces = []
    for status in metrics_dict:
        curr_subjects = metrics_dict[status]
        xs = []
        ys = []
        for subj in curr_subjects:
            index = common_subjects.index(subj)
            xs.append(xs_all[index])
            ys.append(ys_all[index])

        color = cl.scales['8']['qual']['Set1'][list(metrics_dict.keys()).index(status)]
        coordinates = color[4:-1].split(',')
        color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.3) + ')'
        color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

        trace = go.Scatter(
            x=xs,
            y=ys,
            name=status,
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
        traces.append(trace)

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
    )

    fig = go.Figure(data=traces, layout=layout)

    if not os.path.exists(path):
        os.makedirs(path)
    plotly.offline.plot(fig, filename= path + '/pca_' + metrics_key + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, path + '/pca_' + metrics_key + '.png')
    plotly.io.write_image(fig, path + '/pca_' + metrics_key + '.pdf')
