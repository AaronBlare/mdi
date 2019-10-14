from config.otu_counts import load_otu_counts
from config.subjects import load_subject_info, T0_T1_subject_separation
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
import plotly
import plotly.graph_objs as go
from scipy.stats import spearmanr


def plot_random_forest(x, y, title):
    figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'
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
            #range=[min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Predicted adherence",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                )
            ),
            #range=[min(min(x), min(y)) - 5, max(max(x), max(y)) + 5]
        )
    )

    fig = go.Figure(data=trace, layout=layout)

    plotly.offline.plot(fig, filename=figure_file_path + 'rf_' + title + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, figure_file_path + 'rf_' + title + '.png')
    plotly.io.write_image(fig, figure_file_path + 'rf_' + title + '.pdf')

def plot_heatmap(data, names):
    figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'
    trace = go.Heatmap(
        z=[data],
        x=names)

    fig = go.Figure(data=trace)

    plotly.offline.plot(fig, filename=figure_file_path + 'heatmap.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, figure_file_path + 'heatmap.png')
    plotly.io.write_image(fig, figure_file_path + 'heatmap.pdf')


data_file_path = 'D:/Aaron/Bio/NU-Age/Data'

fn_subject_info = data_file_path + '/' + 'correct_subject_info.tsv'
subject_info_dict = load_subject_info(fn_subject_info)
T0_subject_dict, T1_subject_dict = T0_T1_subject_separation(subject_info_dict)
fn_otu_counts = data_file_path + '/' + 'OTUcounts.tsv'
otu_counts = load_otu_counts(fn_otu_counts)

subject_row_dict_T0 = otu_counts.subject_row_dict_T0
subject_row_dict_T1 = otu_counts.subject_row_dict_T1

country_key = 'country'
country_vals = ['Italy', 'UK', 'Holland', 'Poland', 'France']
country_dict = {}

status_key = 'status'
status_vals = ['Subject', 'Control']
status_dict = {}

adherence_key = 'compliance160'
adherence_key_t0 = 'adherence_t0'
adherence_key_t1 = 'adherence_t1'
adherence_dict = {adherence_key_t0: [], adherence_key_t1: []}

common_subjects = list(set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys())))
metadata = {}
subjects_wo_adherence = []
for code in common_subjects:
    index = T0_subject_dict['CODE'].index(code)

    curr_country = T0_subject_dict[country_key][index]
    curr_status = T0_subject_dict[status_key][index]

    curr_adherence_t0 = T0_subject_dict[adherence_key][index]
    curr_adherence_t1 = T1_subject_dict[adherence_key][index]

    if curr_adherence_t0 == '' or curr_adherence_t1 == '':
        subjects_wo_adherence.append(code)
        continue

    metadata[code] = {country_key: curr_country, status_key: curr_status}
    adherence_dict[adherence_key_t0].append(curr_adherence_t0)
    adherence_dict[adherence_key_t1].append(curr_adherence_t1)

    if curr_country in country_dict:
        country_dict[curr_country].append(code)
    else:
        country_dict[curr_country] = [code]

    if curr_status in status_dict:
        status_dict[curr_status].append(code)
    else:
        status_dict[curr_status] = [code]

if len(subjects_wo_adherence) > 0:
    for elem in subjects_wo_adherence:
        common_subjects.remove(elem)

otu_t0 = np.zeros((len(common_subjects), len(list(otu_counts.otu_col_dict.keys()))), dtype=np.float32)
otu_t1 = np.zeros((len(common_subjects), len(list(otu_counts.otu_col_dict.keys()))), dtype=np.float32)

for sub_id, sub in tqdm(enumerate(common_subjects)):
    curr_otu_t0 = otu_counts.normalized_T0[subject_row_dict_T0[sub], :]
    curr_otu_t1 = otu_counts.normalized_T1[subject_row_dict_T1[sub], :]

    otu_t0[sub_id, :] = curr_otu_t0
    otu_t1[sub_id, :] = curr_otu_t1

otu_t0_df = pd.DataFrame(otu_t0, common_subjects, list(otu_counts.otu_col_dict.keys()))
otu_t1_df = pd.DataFrame(otu_t1, common_subjects, list(otu_counts.otu_col_dict.keys()))

clf_t0 = RandomForestRegressor(n_estimators=500)

output_t0 = cross_validate(clf_t0, otu_t0_df, adherence_dict[adherence_key_t0], cv=2, scoring='neg_mean_absolute_error',
                           return_estimator=True)
output_t0_pred = cross_val_predict(clf_t0, otu_t0_df, adherence_dict[adherence_key_t0], cv=2)

plot_random_forest(adherence_dict[adherence_key_t0], output_t0_pred, 'T0')

features_dict_t0 = dict((key, []) for key in list(otu_counts.otu_col_dict.keys()))
for idx, estimator in enumerate(output_t0['estimator']):
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index=list(otu_counts.otu_col_dict.keys()),
                                       columns=['importance']).sort_values('importance', ascending=False)

    features_names = list(feature_importances.index.values)
    features_values = list(feature_importances.values)
    for id in range(0, len(features_names)):
        features_dict_t0[features_names[id]].append(features_values[id][0])

for key in features_dict_t0.keys():
    features_dict_t0[key] = np.mean(features_dict_t0[key])
features_dict_t0 = {k: v for k, v in sorted(features_dict_t0.items(), reverse=True, key=lambda x: x[1])}

num_features = 0
top_features_t0 = []
for key in features_dict_t0.keys():
    if num_features < 75:
        top_features_t0.append(key)
        num_features += 1

clf_t1 = RandomForestRegressor(n_estimators=500)

output_t1 = cross_validate(clf_t1, otu_t1_df, adherence_dict[adherence_key_t1], cv=2, scoring='neg_mean_absolute_error',
                           return_estimator=True)
output_t1_pred = cross_val_predict(clf_t1, otu_t1_df, adherence_dict[adherence_key_t1], cv=2)

plot_random_forest(adherence_dict[adherence_key_t1], output_t1_pred, 'T1')

features_dict_t1 = dict((key, []) for key in list(otu_counts.otu_col_dict.keys()))
for idx, estimator in enumerate(output_t1['estimator']):
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index=list(otu_counts.otu_col_dict.keys()),
                                       columns=['importance']).sort_values('importance', ascending=False)

    features_names = list(feature_importances.index.values)
    features_values = list(feature_importances.values)
    for id in range(0, len(features_names)):
        features_dict_t1[features_names[id]].append(features_values[id][0])

for key in features_dict_t1.keys():
    features_dict_t1[key] = np.mean(features_dict_t1[key])
features_dict_t1 = {k: v for k, v in sorted(features_dict_t1.items(), reverse=True, key=lambda x: x[1])}

num_features = 0
top_features_t1 = []
for key in features_dict_t1.keys():
    if num_features < 75:
        top_features_t1.append(key)
        num_features += 1

top_features_int = list(set(top_features_t0 + top_features_t1))

top_features_art = []
otu_file = data_file_path + '/' + 'otu_random_forest.txt'
f = open(otu_file)
for line in f:
    top_features_art.append(line.replace(' \n', ''))
f.close()

top_features_common_with_art = list(set(top_features_int).intersection(set(top_features_art)))
print('Number of common OTUs: ', str(len(top_features_common_with_art)))

new_df_t0 = otu_t0_df[top_features_int].copy()
new_df_t1 = otu_t1_df[top_features_int].copy()
new_df = new_df_t0.append(new_df_t1)
new_adherence = adherence_dict[adherence_key_t0] + adherence_dict[adherence_key_t1]

corr_coeffs = []
for i in range(0, len(top_features_int)):
    corr_coeff, p_val = spearmanr(list(new_df.iloc[:,i]), new_adherence)
    corr_coeffs.append(corr_coeff)
coeff_range = [min(corr_coeffs), max(corr_coeffs)]

data, names = map(list, zip(*sorted(zip(corr_coeffs, top_features_int), reverse=False)))
plot_heatmap(data, list(range(1, len(top_features_int))))
