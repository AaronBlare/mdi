import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def pipeline_diff_spearman(config):
    common_subjects = config.get_common_subjects_with_adherence()
    otu_counts_delta = config.get_otu_counts_delta(common_subjects)
    adherence_diff, subject_row_adherence_dict = config.get_adherence_diff(common_subjects)
    metrics_dict = init_metrics_dict()

    x = []
    for i, code in enumerate(otu_counts_delta.subject_row_dict):
        curr_adherence = adherence_diff[subject_row_adherence_dict[code]]
        x.append(curr_adherence)

    for otu, col in otu_counts_delta.otu_col_dict.items():
        metrics_dict['otu'].append(otu)
        y = [otu_counts_delta.data[i, col] for i in range(0, len(common_subjects))]
        process_spearman(x, y, metrics_dict)

    reject, pvals_corr, alphacSidak, alphacBonf = multipletests(metrics_dict['p_value'], 0.05, method='fdr_bh')
    metrics_dict['p_value_fdr'] = pvals_corr

    save_table(config, 'OTU_diff_spearman', metrics_dict)


def pipeline_T0_T1_spearman(config):
    common_subjects = config.get_common_subjects_with_adherence()
    common_otus = config.get_common_otus()

    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    common_otu_t0, common_otu_t1, common_otu_col_dict = config.separate_common_otus()

    adherence_key = 'compliance160'
    metadata_t0, obs_dict_t0 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T0')
    metadata_t1, obs_dict_t1 = config.get_target_subject_dicts(common_subjects, [adherence_key], 'T1')

    otu_t0 = np.zeros((len(common_subjects), len(common_otus)), dtype=np.float32)
    otu_t1 = np.zeros((len(common_subjects), len(common_otus)), dtype=np.float32)

    adh_t0 = []
    adh_t1 = []
    for code_id, code in tqdm(enumerate(common_subjects)):
        curr_otu_t0 = common_otu_t0[subject_row_dict_T0[code], :]
        curr_otu_t1 = common_otu_t1[subject_row_dict_T1[code], :]

        otu_t0[code_id, :] = curr_otu_t0
        otu_t1[code_id, :] = curr_otu_t1

        curr_adherence_t0 = metadata_t0[code][adherence_key]
        curr_adherence_t1 = metadata_t1[code][adherence_key]

        adh_t0.append(curr_adherence_t0)
        adh_t1.append(curr_adherence_t1)

    metrics_dict = init_metrics_dict()

    x = adh_t0 + adh_t1

    for otu, col in common_otu_col_dict.items():
        metrics_dict['otu'].append(otu)
        y = np.concatenate((otu_t0[:, col], otu_t1[:, col]), axis=None)
        process_spearman(x, y, metrics_dict)

    reject, pvals_corr, alphacSidak, alphacBonf = multipletests(metrics_dict['p_value'], 0.05, method='fdr_bh')
    metrics_dict['p_value_fdr'] = pvals_corr

    save_table(config, 'T0_T1_spearman', metrics_dict)


def init_metrics_dict():
    metrics_dict = {'otu': [],
                    'rho': [],
                    'p_value': [],
                    'p_value_fdr': []
                    }

    return metrics_dict


def process_spearman(x, y, metrics_dict):

    corr_coeff, p_val = spearmanr(x, y)
    metrics_dict['rho'].append(corr_coeff)
    metrics_dict['p_value'].append(p_val)


def save_table(config, file_name, table_dict):
    path = config.path_out
    fn_xlsx = path + file_name + '.xlsx'
    df = pd.DataFrame(table_dict)
    writer = pd.ExcelWriter(fn_xlsx, engine='xlsxwriter')
    writer.book.use_zip64()
    df.to_excel(writer, index=False)
    writer.save()
