from pcoa.plot import pcoa_plot
from tqdm import tqdm
from skbio.stats.ordination._principal_coordinate_analysis import pcoa
from skbio.stats.distance._base import DistanceMatrix
from skbio.stats.distance._permanova import permanova
from skbio.stats.distance._anosim import anosim
import numpy as np
import pandas as pd

def get_dist_mtx(subjects, row_dict, data):

    distance_mtx = np.zeros((len(subjects), len(subjects)), dtype=np.float32)

    for sub_id_1, sub_1 in tqdm(enumerate(subjects)):
        nut_1 = data[row_dict[sub_1], :]

        for sub_id_2, sub_2 in enumerate(subjects):
            nut_2 = data[row_dict[sub_2], :]

            curr_dist = np.power(np.linalg.norm(nut_1 - nut_2), 2)
            distance_mtx[sub_id_1, sub_id_2] = curr_dist

    check_mtx = np.allclose(distance_mtx, distance_mtx.T, rtol=1e-05, atol=1e-08)
    print(f'Distance matrix is symmetric: {check_mtx}')

    return distance_mtx


def pcoa_pipeline(config, subjects, data_type, time, target_keys):

    config.separate_data(data_type, time)
    row_dict = config.curr_row_dict
    data = config.curr_data

    distance_mtx = get_dist_mtx(subjects, row_dict, data)

    skbio_distance_mtx = DistanceMatrix(distance_mtx, subjects)
    ord_result = pcoa(skbio_distance_mtx)

    metadata, obs_dict = config.get_target_subject_dicts(subjects, target_keys, time)

    metadata_df = pd.DataFrame.from_dict(metadata, orient='index')

    for key, value in obs_dict.items():

        res = permanova(skbio_distance_mtx, metadata_df, key)
        print(f'permanova results for {key}:')
        print(str(res) + '\n')

        res = anosim(skbio_distance_mtx, metadata_df, key)
        print(f'anosim results for {key}:')
        print(str(res) + '\n')

        pcoa_plot(config.path_out, ord_result, subjects, key, obs_dict[key])

    return ord_result