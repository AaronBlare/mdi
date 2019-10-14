import plotly
import colorlover as cl
import plotly.graph_objs as go

def pcoa_plot(ord_result, common_subjects, metrics_key, metrics_dict, prefix ='otu_'):

    coord_matrix = ord_result.samples.values.T
    xs_all = coord_matrix[0]
    ys_all = coord_matrix[1]
    zs_all = coord_matrix[2]

    traces = []
    for status in metrics_dict:
        curr_subjects = metrics_dict[status]
        xs = []
        ys = []
        zs = []
        for subj in curr_subjects:
            index = common_subjects.index(subj)
            xs.append(xs_all[index])
            ys.append(ys_all[index])
            zs.append(zs_all[index])

        color = cl.scales['8']['qual']['Set1'][list(metrics_dict.keys()).index(status)]
        coordinates = color[4:-1].split(',')
        color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.3) + ')'
        color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

        trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
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
        )
    )

    fig = go.Figure(data=traces, layout=layout)

    plotly.offline.plot(fig, filename= prefix +  'pcoa_' + metrics_key + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, prefix +  'pcoa_' + metrics_key + '.png')
    plotly.io.write_image(fig, prefix +  'pcoa_' + metrics_key + '.pdf')
