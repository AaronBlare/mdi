from config.config import Config
import plotly.figure_factory as ff
import plotly.express as px
import plotly
import pandas as pd
import numpy as np
from infrastructure.file_system import get_path


path = get_path()

in_path = path
out_path = path + '/statistics/'

config = Config(in_path, out_path)

subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

status_key = 'status'

metadata_status_t0, obs_dict_status_t0 = config.get_target_subject_dicts(list(subject_row_dict_T0.keys()),
                                                                         [status_key], 'T0')
metadata_status_t1, obs_dict_status_t1 = config.get_target_subject_dicts(list(subject_row_dict_T1.keys()),
                                                                         [status_key], 'T1')

adherence_key = 'compliance160'
common_subjects = config.get_common_subjects_with_adherence()
metadata_ad_t0, obs_dict_ad_t0 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
metadata_ad_t1, obs_dict_ad_t1 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

adherence_key_t0_subject = 't0_subject'
adherence_key_t0_control = 't0_control'
adherence_key_t1_subject = 't1_subject'
adherence_key_t1_control = 't1_control'
adherence_dict = {adherence_key_t0_subject: [], adherence_key_t0_control: [],
                  adherence_key_t1_subject: [], adherence_key_t1_control: []}
adherence_diff_list_subject = []
adherence_diff_list_control = []
adherence_diff = []

subjects_common = []
controls_common = []
for code in common_subjects:
    curr_adherence_t0 = metadata_ad_t0[code][adherence_key]
    curr_adherence_t1 = metadata_ad_t1[code][adherence_key]
    curr_status = metadata_status_t0[code][status_key]

    if curr_status == 'Subject':
        subjects_common.append(code)
        adherence_dict[adherence_key_t0_subject].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_subject].append(curr_adherence_t1)
        adherence_diff_list_subject.append(curr_adherence_t1 - curr_adherence_t0)

    if curr_status == 'Control':
        controls_common.append(code)
        adherence_dict[adherence_key_t0_control].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_control].append(curr_adherence_t1)
        adherence_diff_list_control.append(curr_adherence_t1 - curr_adherence_t0)

    adherence_diff.append(curr_adherence_t1 - curr_adherence_t0)

diff_percentile_val = 4
adherence_diff_percentiles = pd.qcut(adherence_diff, diff_percentile_val, labels=False, retbins=True)

hist_data_t0 = [adherence_dict[adherence_key_t0_subject], adherence_dict[adherence_key_t0_control]]
group_labels_t0 = ['Subject', 'Control']
fig = ff.create_distplot(hist_data_t0, group_labels_t0, show_hist=True, show_rug=False, curve_type='normal')
fig.update_layout(title_text='T0')
plotly.offline.plot(fig, filename=out_path + 'pdf_T0.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'pdf_T0.png')
plotly.io.write_image(fig, out_path + 'pdf_T0.pdf')

hist_data_t1 = [adherence_dict[adherence_key_t1_subject], adherence_dict[adherence_key_t1_control]]
group_labels_t1 = ['Subject', 'Control']
fig = ff.create_distplot(hist_data_t1, group_labels_t1, show_hist=True, show_rug=False, curve_type='normal')
fig.update_layout(title_text='T1')
plotly.offline.plot(fig, filename=out_path + 'pdf_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'pdf_T1.png')
plotly.io.write_image(fig, out_path + 'pdf_T1.pdf')

hist_data = [adherence_dict[adherence_key_t0_subject], adherence_dict[adherence_key_t0_control],
             adherence_dict[adherence_key_t1_subject], adherence_dict[adherence_key_t1_control]]
group_labels = ['Subject T0', 'Control T0', 'Subject T1', 'Control T1']
fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False, curve_type='normal')
plotly.offline.plot(fig, filename=out_path + 'pdf_T0_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'pdf_T0_T1.png')
plotly.io.write_image(fig, out_path + 'pdf_T0_T1.pdf')

hist_data_adh = [adherence_diff_list_subject, adherence_diff_list_control]
group_labels = ['Subject', 'Control']
fig = ff.create_distplot(hist_data_adh, group_labels, show_hist=True, show_rug=False, curve_type='normal')
fig.add_scatter(
    x=[adherence_diff_percentiles[1][diff_percentile_val - 1], adherence_diff_percentiles[1][diff_percentile_val - 1]],
    y=[0, 0.1], mode="lines",
    marker=dict(color="Red"), line=dict(width=5), name='Last quartile')
fig.add_scatter(x=[adherence_diff_percentiles[1][1], adherence_diff_percentiles[1][1]], y=[0, 0.1], mode="lines",
                marker=dict(color="Green"), line=dict(width=5), name='First quartile')
plotly.offline.plot(fig, filename=out_path + 'pdf_adherence_diff.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'pdf_adherence_diff.png')
plotly.io.write_image(fig, out_path + 'pdf_adherence_diff.pdf')

combined_adherence_t0 = adherence_dict[adherence_key_t0_subject] + adherence_dict[adherence_key_t0_control]
combined_status_t0 = ['Subject', ] * len(adherence_dict[adherence_key_t0_subject]) + \
                     ['Control', ] * len(adherence_dict[adherence_key_t0_control])
data_t0 = pd.DataFrame(np.column_stack([combined_adherence_t0, combined_status_t0]),
                       columns=['adherence', 'status'])
fig = px.histogram(data_t0, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7,
                   nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=out_path + 'ex_pdf_T0.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'ex_pdf_T0.png')
plotly.io.write_image(fig, out_path + 'ex_pdf_T0.pdf')

combined_adherence_t1 = adherence_dict[adherence_key_t1_subject] + adherence_dict[adherence_key_t1_control]
combined_status_t1 = ['Subject', ] * len(adherence_dict[adherence_key_t1_subject]) + \
                     ['Control', ] * len(adherence_dict[adherence_key_t1_control])
data_t1 = pd.DataFrame(np.column_stack([combined_adherence_t1, combined_status_t1]),
                       columns=['adherence', 'status'])
fig = px.histogram(data_t1, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7,
                   nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=out_path + 'ex_pdf_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'ex_pdf_T1.png')
plotly.io.write_image(fig, out_path + 'ex_pdf_T1.pdf')

combined_adherence = adherence_dict[adherence_key_t0_subject] + adherence_dict[adherence_key_t0_control] + \
                     adherence_dict[adherence_key_t1_subject] + adherence_dict[adherence_key_t1_control]
combined_status = ['Subject T0', ] * len(adherence_dict[adherence_key_t0_subject]) + \
                  ['Control T0', ] * len(adherence_dict[adherence_key_t0_control]) + \
                  ['Subject T1', ] * len(adherence_dict[adherence_key_t1_subject]) + \
                  ['Control T1', ] * len(adherence_dict[adherence_key_t1_control])
data = pd.DataFrame(np.column_stack([combined_adherence, combined_status]),
                    columns=['adherence', 'status'])
fig = px.histogram(data, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7,
                   nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=out_path + 'ex_pdf_T0_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'ex_pdf_T0_T1.png')
plotly.io.write_image(fig, out_path + 'ex_pdf_T0_T1.pdf')

combined_adherence_diff = adherence_diff_list_subject + adherence_diff_list_control
combined_status = ['Subject', ] * len(adherence_diff_list_subject) + \
                  ['Control', ] * len(adherence_diff_list_control)
data = pd.DataFrame(np.column_stack([combined_adherence_diff, combined_status]),
                    columns=['adherence diff', 'status'])
fig = px.histogram(data, x="adherence diff", color="status", histnorm="probability density", marginal="box",
                   opacity=0.7, nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=out_path + 'ex_pdf_adherence_diff.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, out_path + 'ex_pdf_adherence_diff.png')
plotly.io.write_image(fig, out_path + 'ex_pdf_adherence_diff.pdf')
