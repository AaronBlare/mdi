from config.config import Config
import numpy as np
import plotly
import plotly.graph_objects as go
from routines.plot import get_margin
from infrastructure.file_system import get_path
from infrastructure.load.table import load_table_dict_xlsx
import os
from scipy.stats import spearmanr
from routines.plot import cmocean_to_plotly
import cmocean
from statsmodels.stats.multitest import multipletests

def plot_heatmaps(xs, ys, rhos, p_values, time):

    layout = go.Layout(
        margin=get_margin(),
        autosize=True,
        showlegend=False,
        yaxis=dict(
            type='category',
            showgrid=True,
            showline=True,
            mirror='ticks',
            titlefont=dict(
                family='Arial',
                color='black',
                size=2,
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                family='Arial',
                color='black',
                size=2
            ),
            exponentformat='e',
            showexponent='all'
        )
    )

    passed, p_values_corr, _, _ = multipletests(p_values.flatten(), 0.05, method='fdr_bh')
    passed.shape = (len(ys), len(xs))
    passed = passed.astype(int)
    p_values_corr.shape = (len(ys), len(xs))

    trace = go.Heatmap(
        z=rhos,
        x=xs,
        y=ys,
        colorscale=balance)
    fig = go.Figure(data=trace, layout=layout)
    plotly.offline.plot(fig, filename=out_path + '/rhos_' + time + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, out_path + '/rhos_' + time + '.png')
    plotly.io.write_image(fig, out_path + '/rhos_' + time + '.pdf')

    trace = go.Heatmap(
        z=-np.log10(p_values),
        x=xs,
        y=ys,
        colorscale=dense_inv)
    fig = go.Figure(data=trace, layout=layout)
    plotly.offline.plot(fig, filename=out_path + '/p_values_' + time + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, out_path + '/p_values_' + time + '.png')
    plotly.io.write_image(fig, out_path + '/p_values_' + time + '.pdf')

    trace = go.Heatmap(
        z=-np.log10(p_values_corr),
        x=xs,
        y=ys,
        colorscale=dense_inv)
    fig = go.Figure(data=trace, layout=layout)
    plotly.offline.plot(fig, filename=out_path + '/p_values_corr_' + time + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, out_path + '/p_values_corr_' + time + '.png')
    plotly.io.write_image(fig, out_path + '/p_values_corr_' + time + '.pdf')

    trace = go.Heatmap(
        z=passed,
        x=xs,
        y=ys)
    fig = go.Figure(data=trace, layout=layout)
    plotly.offline.plot(fig, filename=out_path + '/passed_' + time + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, out_path + '/passed_' + time + '.png')
    plotly.io.write_image(fig, out_path + '/passed_' + time + '.pdf')

balance = cmocean_to_plotly(cmocean.cm.balance, 100)
dense_inv = cmocean_to_plotly(cmocean.cm.dense, 10)

path = get_path()

in_path = path
out_path = path + '/spearman/supp_fig_11'
if not os.path.isdir(out_path):
    os.makedirs(out_path)

f = open(path + '/original/cytokines.txt')
cytokines = f.read().splitlines()
f.close()

config = Config(in_path, out_path)
common_subjects = config.get_common_subjects_with_adherence_and_cytokines(cytokines)

xs = cytokines

regressor_dict = load_table_dict_xlsx(path + '/rf_regressor/top_OTU_spearman.xlsx')

otus = np.array(regressor_dict['otu'])
otus_rho = np.array(regressor_dict['rho'])
otus_p_value_fdr = np.array(regressor_dict['p_value_fdr'])

otus_order = np.argsort(otus_rho)
otus = otus[otus_order]
otus_rho = otus_rho[otus_order]
otus_p_value_fdr = otus_p_value_fdr[otus_order]

ys = otus

common_subjects_entire = common_subjects + common_subjects
subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1
common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()
common_otu_entire = np.concatenate((common_otu_t0, common_otu_t1), axis=0)

subject_ids_t0 = np.array([subject_row_dict_T0[x] for x in common_subjects], dtype=int)
subject_ids_t1 = np.array([subject_row_dict_T1[x] for x in common_subjects], dtype=int)
subject_ids_t1_for_entire_ids = np.array([subject_row_dict_T1[x] + common_otu_t0.shape[0] for x in common_subjects])
subject_ids_entire = np.concatenate((subject_ids_t0, subject_ids_t1_for_entire_ids), axis=None)

subject_ids_cytokine_t0 = np.array([config.cytokines_T0['CODE'].index(code) for code in common_subjects])
subject_ids_cytokine_t1 = np.array([config.cytokines_T1['CODE'].index(code) for code in common_subjects])

rho_t0 = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)
rho_t1 = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)
rho_entire = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)

p_value_t0 = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)
p_value_t1 = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)
p_value_entire = np.zeros(shape=(len(ys), len(xs)), dtype=np.float32)

for y_id, y in enumerate(ys): # otu
    for x_id, x in enumerate(xs): # cytokine

        otus_cols = [common_otu_col_dict[y]]

        corr_y_t0 = common_otu_t0[np.ix_(subject_ids_t0, otus_cols)]
        corr_y_t1 = common_otu_t1[np.ix_(subject_ids_t1, otus_cols)]
        corr_y_entire = common_otu_entire[np.ix_(subject_ids_entire, otus_cols)]

        corr_x_t0 = np.array(np.array(config.cytokines_T0[x])[subject_ids_cytokine_t0], dtype=np.float32)
        corr_x_t1 = np.array(np.array(config.cytokines_T1[x])[subject_ids_cytokine_t1], dtype=np.float32)
        corr_x_entire = np.concatenate((corr_x_t0, corr_x_t1), axis=None)

        rho, p_value = spearmanr(corr_x_t0, corr_y_t0)
        rho_t0[y_id, x_id] = rho
        p_value_t0[y_id, x_id] = p_value

        rho, p_value = spearmanr(corr_x_t1, corr_y_t1)
        rho_t1[y_id, x_id] = rho
        p_value_t1[y_id, x_id] = p_value

        rho, p_value = spearmanr(corr_x_entire, corr_y_entire)
        rho_entire[y_id, x_id] = rho
        p_value_entire[y_id, x_id] = p_value

plot_heatmaps(xs, ys, rho_t0, p_value_t0, 't0')
plot_heatmaps(xs, ys, rho_t1, p_value_t1, 't1')
plot_heatmaps(xs, ys, rho_entire, p_value_entire, 'entire')