from config.otu_counts import load_otu_counts
from config.nutrition import load_nutrition
from config.food_groups import load_food_groups
from config.subjects import load_subject_info, T0_T1_subject_separation
import pandas as pd
from pcoa.pipeline import pcoa_pipeline
from scipy.spatial import procrustes
import numpy as np

path = 'D:/YandexDisk/Work/nuage'

fn_subject_info = path + '/' + 'correct_subject_info.tsv'
subject_info_dict = load_subject_info(fn_subject_info)
T0_subject_dict, T1_subject_dict = T0_T1_subject_separation(subject_info_dict)

fn_food_groups = path + '/' + 'food_groups_long.tsv'
food_groups = load_food_groups(fn_food_groups, norm=1)

fn_nutrition = path + '/' + 'nutrition_long.tsv'
nutrition = load_nutrition(fn_nutrition, norm=1)

fn_otu_counts = path + '/' + 'OTUcounts.tsv'
otu_counts = load_otu_counts(fn_otu_counts, norm=0)

food_subj_row_dict_T0 = food_groups.subject_row_dicts['T0']
food_subj_row_dict_T1 = food_groups.subject_row_dicts['T1']

nut_subj_row_dict_T0 = nutrition.subject_row_dicts['T0']
nut_subj_row_dict_T1 = nutrition.subject_row_dicts['T1']

subject_row_dict_T0 = otu_counts.subject_row_dict_T0
subject_row_dict_T1 = otu_counts.subject_row_dict_T1

country_key = 'country'
country_vals = ['Italy', 'UK', 'Holland', 'Poland', 'France']
country_dict = {}

status_key = 'status'
status_vals = ['Subject', 'Control']
status_dict = {}

common_subjects_food = list(set(food_subj_row_dict_T0.keys()).intersection(set(food_subj_row_dict_T1.keys())))
common_subjects_nutrition = list(set(nut_subj_row_dict_T0.keys()).intersection(set(nut_subj_row_dict_T1.keys())))
common_subjects_otu = set(subject_row_dict_T0.keys()).intersection(set(subject_row_dict_T1.keys()))

common_subjects = list(common_subjects_otu.intersection(common_subjects_nutrition).intersection(common_subjects_food))
print(f'\nNumber of common subjects: {len(common_subjects)}')
metadata = {}
for code in common_subjects:
    index = T0_subject_dict['CODE'].index(code)

    curr_country = T0_subject_dict[country_key][index]
    curr_status = T0_subject_dict[status_key][index]

    metadata[code] = {country_key: curr_country, status_key: curr_status}

    if curr_country in country_dict:
        country_dict[curr_country].append(code)
    else:
        country_dict[curr_country] = [code]

    if curr_status in status_dict:
        status_dict[curr_status].append(code)
    else:
        status_dict[curr_status] = [code]

metadata_df = pd.DataFrame.from_dict(metadata, orient='index')

otu_res = pcoa_pipeline(
    common_subjects,
    otu_counts.subject_row_dict_T0,
    otu_counts.normalized_T0,
    metadata_df,
    status_dict,
    country_dict,
    status_key,
    country_key,
    prefix='otu_'
)

nut_res = pcoa_pipeline(
    common_subjects,
    nutrition.subject_row_dicts['T0'],
    nutrition.nutrition_data_dict['T0'],
    metadata_df,
    status_dict,
    country_dict,
    status_key,
    country_key,
    prefix='nutrition_'
)

food_res = pcoa_pipeline(
    common_subjects,
    food_groups.subject_row_dicts['T0'],
    food_groups.food_groups_data_dict['T0'],
    metadata_df,
    status_dict,
    country_dict,
    status_key,
    country_key,
    prefix='food_groups_'
)

otu_pcoa_coord_matrix = otu_res.samples.values.T
otu_pcoa_data = np.array([otu_pcoa_coord_matrix[0], otu_pcoa_coord_matrix[1], otu_pcoa_coord_matrix[2]]).T

nut_pcoa_coord_matrix = nut_res.samples.values.T
nut_pcoa_data = np.array([nut_pcoa_coord_matrix[0], nut_pcoa_coord_matrix[1], nut_pcoa_coord_matrix[2]]).T

food_pcoa_coord_matrix = food_res.samples.values.T
food_pcoa_data = np.array([food_pcoa_coord_matrix[0], food_pcoa_coord_matrix[1], food_pcoa_coord_matrix[2]]).T


mtx1, mtx2, disparity = procrustes(nut_pcoa_data, food_pcoa_data)
print(f'disparity(nutrition, food_groups): {disparity}')

a = 0
