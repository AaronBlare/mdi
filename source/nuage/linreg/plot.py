import statsmodels.api as sm
import numpy as np
import plotly.graph_objs as go
import plotly


def pipeline_plot(config, otu_file, plot_type):
    otus = read_otu_file(otu_file)
    common_subjects = config.get_common_subjects_with_adherence()
    if plot_type == 'subject':
        status_key = 'status'
        subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
        metadata_status, obs_dict_status = config.get_target_subject_dicts(list(subject_row_dict_T0.keys()),
                                                                           [status_key], 'T0')
        otu_counts_delta = config.get_otu_counts_delta(common_subjects)
        adherence_diff, subject_row_adherence_dict = config.get_adherence_diff(common_subjects)
        x = [[], []]
        for i, code in enumerate(otu_counts_delta.subject_row_dict):
            curr_adherence = adherence_diff[subject_row_adherence_dict[code]]
            if code in obs_dict_status[status_key]['Subject']:
                x[0].append(curr_adherence)
            elif code in obs_dict_status[status_key]['Control']:
                x[1].append(curr_adherence)

        for otu in otus:
            y = [[], []]
            col = otu_counts_delta.otu_col_dict[otu]
            for i, code in enumerate(common_subjects):
                if code in obs_dict_status[status_key]['Subject']:
                    y[0].append(otu_counts_delta.data[i, col])
                elif code in obs_dict_status[status_key]['Control']:
                    y[1].append(otu_counts_delta.data[i, col])
            plot_linreg(config, x, y, otu, plot_type)

    elif plot_type == 'all':
        otu_counts_delta = config.get_otu_counts_delta(common_subjects)
        adherence_diff, subject_row_adherence_dict = config.get_adherence_diff(common_subjects)
        x = []
        for i, code in enumerate(otu_counts_delta.subject_row_dict):
            curr_adherence = adherence_diff[subject_row_adherence_dict[code]]
            x.append(curr_adherence)

        for otu in otus:
            col = otu_counts_delta.otu_col_dict[otu]
            y = [otu_counts_delta.data[i, col] for i in range(0, len(common_subjects))]
            plot_linreg(config, x, y, otu, plot_type)


def plot_linreg(config, xs, ys, name, plot_type):
    fig = go.Figure()

    if plot_type == 'subject':
        x = xs[0]
        y = ys[0]
    else:
        x = xs
        y = ys

    fig.add_scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=4,
            color='Red',
            line=dict(
                width=1,
                color='Red',
            )
        ),
        name='Subject'
    )

    if plot_type == 'subject':
        x = xs[1]
        y = ys[1]

        fig.add_scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=4,
                color='Blue',
                line=dict(
                    width=1,
                    color='Blue',
                )
            ),
            name='Control'
        )

    if plot_type == 'subject':
        x_common = xs[0] + xs[1]
        y_common = ys[0] + ys[1]
    else:
        x_common = xs
        y_common = ys
    # Linear regression
    x = sm.add_constant(x_common)
    y = y_common
    results = sm.OLS(y, x).fit()
    intercept = results.params[0]
    slope = results.params[1]

    # Adding regression line
    x_min = np.min(x_common)
    x_max = np.max(x_common)
    y_min = slope * x_min + intercept
    y_max = slope * x_max + intercept

    fig.add_scatter(
        x=[x_min, x_max],
        y=[y_min, y_max],
        mode='lines',
        line=dict(
            width=6,
            color='Red'
        ),
        showlegend=False
    )

    title = name
    y_title = 'OTU difference'
    x_title = 'Adherence difference'
    fig.update_layout(go.Layout(
        title=dict(
            text=title,
            font=dict(
                family='Arial',
                size=33,
            )
        ),
        autosize=True,
        margin=go.layout.Margin(
            l=110,
            r=10,
            b=80,
            t=85,
            pad=0
        ),
        barmode='overlay',
        xaxis=dict(
            title=x_title,
            showgrid=True,
            showline=True,
            mirror='ticks',
            titlefont=dict(
                family='Arial',
                size=33,
                color='black'
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                family='Arial',
                size=30,
                color='black'
            )
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            showline=True,
            mirror='ticks',
            titlefont=dict(
                family='Arial',
                size=33,
                color='black'
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                family='Arial',
                size=30,
                color='black'
            )
        )
    ))

    if plot_type == 'subject':
        fig.update_layout(showlegend=True)
        name += '_subject'
    else:
        fig.update_layout(showlegend=False)

    path = config.path_out
    plotly.io.write_image(fig, path + name + '.png')


def read_otu_file(otu_file):
    otus = []
    f = open(otu_file)
    for line in f:
        otus.append(line.replace('\n', ''))
    f.close()

    return otus
