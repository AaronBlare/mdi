from pca.plot import pca_plot
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_pipeline(config, subjects, data_type, time, target_keys):

    config.separate_data(data_type, time)

    col_dict = config.curr_col_dict
    row_dict = config.curr_raw_dict
    raw_data = config.curr_data

    data = np.zeros(shape=(len(subjects), raw_data.shape[1]))
    for sub_id_1, sub_1 in tqdm(enumerate(subjects)):
        data[sub_id_1, :] = raw_data[row_dict[sub_1], :]

    data = StandardScaler().fit_transform(data)

    columns = list(col_dict.keys())
    data_df = pd.DataFrame(data, columns=columns)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(data_df)

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    metadata, obs_dict = config.get_target_subject_dicts(subjects, target_keys, time)

    for key, value in obs_dict.items():
        pca_plot(config.path_out, pc, subjects, key, obs_dict[key])
