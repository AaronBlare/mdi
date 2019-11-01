import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro, kstest, normaltest
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson

def pipeline_linreg(config):
    subject_row_dict_T0 = config.otu_counts.subject_row_dict_T0
    subject_row_dict_T1 = config.otu_counts.subject_row_dict_T1

    config.separate_data('otu_counts', 'T0')
    otu_t0 = config.curr_data
    otu_t0_col_dict = config.curr_col_dict

