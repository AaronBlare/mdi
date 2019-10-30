from config.otu_counts import load_otu_counts
from config.subjects import load_subject_info, T0_T1_subject_separation
import plotly.figure_factory as ff
import plotly.express as px
import plotly
import pandas as pd
import numpy as np

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'

fn_subject_info = data_file_path + '/' + 'correct_subject_info.tsv'
subject_info_dict = load_subject_info(fn_subject_info)
T0_subject_dict, T1_subject_dict = T0_T1_subject_separation(subject_info_dict)
fn_otu_counts = data_file_path + '/' + 'OTUcounts.tsv'
otu_counts = load_otu_counts(fn_otu_counts)

subject_row_dict_T0 = otu_counts.subject_row_dict_T0
subject_row_dict_T1 = otu_counts.subject_row_dict_T1

adherence_key = 'compliance160'
adherence_key_t0_subject = 'adherence_t0_subject'
adherence_key_t0_control = 'adherence_t0_control'
adherence_key_t1_subject = 'adherence_t1_subject'
adherence_key_t1_control = 'adherence_t1_control'
adherence_dict = {adherence_key_t0_subject: [], adherence_key_t0_control: [],
                  adherence_key_t1_subject: [], adherence_key_t1_control: []}

common_subjects = list(set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys())))
subjects_wo_adherence = []

for code in common_subjects:
    index_t0 = T0_subject_dict['CODE'].index(code)
    index_t1 = T1_subject_dict['CODE'].index(code)

    curr_adherence_t0 = T0_subject_dict[adherence_key][index_t0]
    curr_adherence_t1 = T1_subject_dict[adherence_key][index_t1]

    if curr_adherence_t0 == '' or curr_adherence_t1 == '':
        subjects_wo_adherence.append(code)
        continue

    if T0_subject_dict['status'][index_t0] == 'Subject':
        adherence_dict[adherence_key_t0_subject].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_subject].append(curr_adherence_t1)

    if T0_subject_dict['status'][index_t0] == 'Control':
        adherence_dict[adherence_key_t0_control].append(curr_adherence_t0)
        adherence_dict[adherence_key_t1_control].append(curr_adherence_t1)

if len(subjects_wo_adherence) > 0:
    for elem in subjects_wo_adherence:
        common_subjects.remove(elem)

figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'

hist_data_t0 = [adherence_dict[adherence_key_t0_subject], adherence_dict[adherence_key_t0_control]]
group_labels_t0 = ['Subject', 'Control']
fig = ff.create_distplot(hist_data_t0, group_labels_t0, show_hist=True, show_rug=False, curve_type='normal')
fig.update_layout(title_text='T0')
plotly.offline.plot(fig, filename=figure_file_path + 'pdf_T0.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'pdf_T0.png')
plotly.io.write_image(fig, figure_file_path + 'pdf_T0.pdf')

hist_data_t1 = [adherence_dict[adherence_key_t1_subject], adherence_dict[adherence_key_t1_control]]
group_labels_t1 = ['Subject', 'Control']
fig = ff.create_distplot(hist_data_t1, group_labels_t1, show_hist=True, show_rug=False, curve_type='normal')
fig.update_layout(title_text='T1')
plotly.offline.plot(fig, filename=figure_file_path + 'pdf_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'pdf_T1.png')
plotly.io.write_image(fig, figure_file_path + 'pdf_T1.pdf')

hist_data = [adherence_dict[adherence_key_t0_subject], adherence_dict[adherence_key_t0_control],
             adherence_dict[adherence_key_t1_subject], adherence_dict[adherence_key_t1_control]]
group_labels = ['Subject T0', 'Control T0', 'Subject T1', 'Control T1']
fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False, curve_type='normal')
plotly.offline.plot(fig, filename=figure_file_path + 'pdf_T0_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'pdf_T0_T1.png')
plotly.io.write_image(fig, figure_file_path + 'pdf_T0_T1.pdf')

combined_adherence_t0 = adherence_dict[adherence_key_t0_subject] + adherence_dict[adherence_key_t0_control]
combined_status_t0 = ['Subject',] * len(adherence_dict[adherence_key_t0_subject]) + ['Control',] * len(adherence_dict[adherence_key_t0_control])
data_t0 = pd.DataFrame(np.column_stack([combined_adherence_t0, combined_status_t0]),
                       columns=['adherence', 'status'])
fig = px.histogram(data_t0, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7, nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=figure_file_path + 'ex_pdf_T0.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T0.png')
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T0.pdf')


combined_adherence_t1 = adherence_dict[adherence_key_t1_subject] + adherence_dict[adherence_key_t1_control]
combined_status_t1 = ['Subject',] * len(adherence_dict[adherence_key_t1_subject]) + ['Control',] * len(adherence_dict[adherence_key_t1_control])
data_t1 = pd.DataFrame(np.column_stack([combined_adherence_t1, combined_status_t1]),
                       columns=['adherence', 'status'])
fig = px.histogram(data_t1, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7, nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=figure_file_path + 'ex_pdf_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T1.png')
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T1.pdf')


combined_adherence = adherence_dict[adherence_key_t0_subject] + adherence_dict[adherence_key_t0_control] + \
                     adherence_dict[adherence_key_t1_subject] + adherence_dict[adherence_key_t1_control]
combined_status = ['Subject T0',] * len(adherence_dict[adherence_key_t0_subject]) + ['Control T0',] * len(adherence_dict[adherence_key_t0_control]) + \
                  ['Subject T1',] * len(adherence_dict[adherence_key_t1_subject]) + ['Control T1',] * len(adherence_dict[adherence_key_t1_control])
data = pd.DataFrame(np.column_stack([combined_adherence, combined_status]),
                       columns=['adherence', 'status'])
fig = px.histogram(data, x="adherence", color="status", histnorm="probability density", marginal="box", opacity=0.7, nbins=200, barmode="overlay")
plotly.offline.plot(fig, filename=figure_file_path + 'ex_pdf_T0_T1.html', auto_open=False, show_link=True)
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T0_T1.png')
plotly.io.write_image(fig, figure_file_path + 'ex_pdf_T0_T1.pdf')
