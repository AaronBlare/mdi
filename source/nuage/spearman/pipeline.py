import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def pipeline_spearman(config):
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
