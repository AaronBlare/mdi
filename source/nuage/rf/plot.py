import plotly
import plotly.graph_objs as go


def plot_scatter(x, y, title, suffix, figure_file_path):
    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        marker=dict(
            size=8,
            line=dict(
                width=0.5
            ),
            opacity=0.8
        )
    )
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        title=go.layout.Title(
            text=title,
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Number of OTUs",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                )
            ),
            #range=[min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=title,
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                )
            ),
            #range=[min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
        )
    )

    fig = go.Figure(data=trace, layout=layout)

    plotly.offline.plot(fig, filename=figure_file_path + '/scatter_' + suffix + '.html', auto_open=False, show_link=True)
    # plotly.io.write_image(fig, figure_file_path + '/scatter_' + suffix + '.png')
    # plotly.io.write_image(fig, figure_file_path + '/scatter_' + suffix + '.pdf')


def plot_random_forest(x, y, title, is_equal_range, figure_file_path):
    if is_equal_range:
        xrange = [min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
        yrange = [min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
    else:
        xrange = [min(x) - 1, max(x) + 1]
        yrange = [min(y) - 1, max(y) + 1]
    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            line=dict(
                width=0.5
            ),
            opacity=0.8
        )
    )
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        title=go.layout.Title(
            text=title,
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Actual adherence",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                )
            ),
            range=xrange
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Predicted adherence",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                )
            ),
            range=yrange
        )
    )

    fig = go.Figure(data=trace, layout=layout)

    if is_equal_range:
        title = title + '_eq'

    plotly.offline.plot(fig, filename=figure_file_path + 'rf_' + title + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, figure_file_path + 'rf_' + title + '.png')
    plotly.io.write_image(fig, figure_file_path + 'rf_' + title + '.pdf')


def plot_heatmap(data, names, figure_file_path):
    trace = go.Heatmap(
        z=[data],
        x=names)

    fig = go.Figure(data=trace)

    plotly.offline.plot(fig, filename=figure_file_path + 'heatmap.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, figure_file_path + 'heatmap.png')
    plotly.io.write_image(fig, figure_file_path + 'heatmap.pdf')


def plot_hist(data, names, colors, suffix, figure_file_path):
    fig = go.Figure(go.Bar(
        x=data,
        y=names,
        orientation='h',
        marker_color=colors
    ))
    fig.update_yaxes(
        tickfont=dict(size=10)
    )
    fig.update_layout(width=700,
                      height=1000)

    plotly.offline.plot(fig, filename=figure_file_path + suffix + '_hist.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, figure_file_path + suffix + '_hist.png')
    plotly.io.write_image(fig, figure_file_path + suffix + '_hist.pdf')


def plot_box(data_dict, names, figure_file_path, title):
    fig = go.Figure()
    for name in names:
        data = data_dict[name]
        fig.add_trace(go.Box(x=data, name=name))

    plotly.offline.plot(fig, filename=figure_file_path + '/' + title + '_boxplot.html', auto_open=False, show_link=True)
    # plotly.io.write_image(fig, figure_file_path + 'boxplot.png')
    # plotly.io.write_image(fig, figure_file_path + 'boxplot.pdf')

