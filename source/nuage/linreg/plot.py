import statsmodels.api as sm
import numpy as np
import plotly.graph_objs as go
import plotly


def plot_linreg(config, xs, ys, name):
    fig = go.Figure()

    fig.add_scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(
            size=4,
            color='Red',
            line=dict(
                width=1,
                color='Red',
            )
        ),
    )

    # Linear regression
    x = sm.add_constant(xs)
    y = ys
    results = sm.OLS(y, x).fit()
    intercept = results.params[0]
    slope = results.params[1]

    # Adding regression line
    x_min = np.min(xs)
    x_max = np.max(xs)
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
    fig.update_layout(showlegend=False)

    path = config.path_out
    plotly.io.write_image(fig, path + name + '.png')
