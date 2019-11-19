import plotly.graph_objs as go


def get_axis(title):
    axis = dict(
        title=title,
        showgrid=True,
        showline=True,
        mirror='ticks',
        titlefont=dict(
            family='Arial',
            color='black'
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            family='Arial',
            color='black'
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
