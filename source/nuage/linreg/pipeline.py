import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import shapiro, kstest, normaltest
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson
from tqdm import tqdm


def pipeline_diff_linreg(config):
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
        process_linreg(x, y, metrics_dict)

    save_table(config, 'diff_linreg', metrics_dict)


def pipeline_T0_T1_linreg(config):
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
        process_linreg(x, y, metrics_dict)

    save_table(config, 'T0_T1_linreg', metrics_dict)


def init_metrics_dict():
    metrics_dict = {'otu': [],
                    'R2': [],
                    'R2_adj': [],
                    'f_stat': [],
                    'prob(f_stat)': [],
                    'log_likelihood': [],
                    'AIC': [],
                    'BIC': [],
                    'omnibus': [],
                    'prob(omnibus)': [],
                    'skew': [],
                    'kurtosis': [],
                    'durbin_watson': [],
                    'jarque_bera': [],
                    'prob(jarque_bera)': [],
                    'cond_no': [],
                    'normality_p_value_shapiro': [],
                    'normality_p_value_ks_wo_params': [],
                    'normality_p_value_ks_with_params': [],
                    'normality_p_value_dagostino': [],
                    'intercept': [],
                    'slope': [],
                    'intercept_std': [],
                    'slope_std': [],
                    'intercept_p_value': [],
                    'slope_p_value': []
                    }

    return metrics_dict


def process_linreg(x, y, metrics_dict):
    x = sm.add_constant(x)

    results = sm.OLS(y, x).fit()

    residuals = results.resid

    jb, jbpv, skew, kurtosis = jarque_bera(results.wresid)
    omni, omnipv = omni_normtest(results.wresid)

    res_mean = np.mean(residuals)
    res_std = np.std(residuals)

    _, normality_p_value_shapiro = shapiro(residuals)
    _, normality_p_value_ks_wo_params = kstest(residuals, 'norm')
    _, normality_p_value_ks_with_params = kstest(residuals, 'norm', (res_mean, res_std))
    _, normality_p_value_dagostino = normaltest(residuals)

    metrics_dict['R2'].append(results.rsquared)
    metrics_dict['R2_adj'].append(results.rsquared_adj)
    metrics_dict['f_stat'].append(results.fvalue)
    metrics_dict['prob(f_stat)'].append(results.f_pvalue)
    metrics_dict['log_likelihood'].append(results.llf)
    metrics_dict['AIC'].append(results.aic)
    metrics_dict['BIC'].append(results.bic)
    metrics_dict['omnibus'].append(omni)
    metrics_dict['prob(omnibus)'].append(omnipv)
    metrics_dict['skew'].append(skew)
    metrics_dict['kurtosis'].append(kurtosis)
    metrics_dict['durbin_watson'].append(durbin_watson(results.wresid))
    metrics_dict['jarque_bera'].append(jb)
    metrics_dict['prob(jarque_bera)'].append(jbpv)
    metrics_dict['cond_no'].append(results.condition_number)
    metrics_dict['normality_p_value_shapiro'].append(normality_p_value_shapiro)
    metrics_dict['normality_p_value_ks_wo_params'].append(normality_p_value_ks_wo_params)
    metrics_dict['normality_p_value_ks_with_params'].append(normality_p_value_ks_with_params)
    metrics_dict['normality_p_value_dagostino'].append(normality_p_value_dagostino)
    metrics_dict['intercept'].append(results.params[0])
    metrics_dict['slope'].append(results.params[1])
    metrics_dict['intercept_std'].append(results.bse[0])
    metrics_dict['slope_std'].append(results.bse[1])
    metrics_dict['intercept_p_value'].append(results.pvalues[0])
    metrics_dict['slope_p_value'].append(results.pvalues[1])


def save_table(config, file_name, table_dict):
    path = config.path_out
    fn_xlsx = path + file_name + '.xlsx'
    df = pd.DataFrame(table_dict)
    writer = pd.ExcelWriter(fn_xlsx, engine='xlsxwriter')
    writer.book.use_zip64()
    df.to_excel(writer, index=False)
    writer.save()
