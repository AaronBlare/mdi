from pcoa.plot import pcoa_plot
from tqdm import tqdm
from skbio.stats.ordination._principal_coordinate_analysis import pcoa
from skbio.stats.distance._base import DistanceMatrix
from skbio.stats.distance._permanova import permanova
from skbio.stats.distance._anosim import anosim
import numpy as np


def pcoa_pipeline(common_subjects, subject_row_dict, data, metadata_df, status_dict, country_dict, status_key, country_key, prefix):

    distance_mtx = np.zeros((len(common_subjects), len(common_subjects)), dtype=np.float32)

    for sub_id_1, sub_1 in tqdm(enumerate(common_subjects)):
        nut_1 = data[subject_row_dict[sub_1], :]

        for sub_id_2, sub_2 in enumerate(common_subjects):
            nut_2 = data[subject_row_dict[sub_2], :]

            curr_dist = np.power(np.linalg.norm(nut_1 - nut_2), 2)
            distance_mtx[sub_id_1, sub_id_2] = curr_dist

    check_mtx = np.allclose(distance_mtx, distance_mtx.T, rtol=1e-05, atol=1e-08)
    print(f'Is distance matrix symmetric: {check_mtx}')

    skbio_distance_matrix = DistanceMatrix(distance_mtx, common_subjects)
    ord_result = pcoa(skbio_distance_matrix)

    permanova_status = permanova(skbio_distance_matrix, metadata_df, status_key)
    print('permanova results for status:\n' + str(permanova_status) + '\n\n')
    permanova_country = permanova(skbio_distance_matrix, metadata_df, country_key)
    print('permanova results for country:\n' + str(permanova_country) + '\n\n')

    anosim_status = anosim(skbio_distance_matrix, metadata_df, status_key)
    print('anosim results for status:\n' + str(anosim_status) + '\n\n')
    anosim_country = anosim(skbio_distance_matrix, metadata_df, country_key)
    print('anosim results for country:\n' + str(anosim_country) + '\n\n')

    pcoa_plot(ord_result, common_subjects, status_key, status_dict, prefix)
    pcoa_plot(ord_result, common_subjects, country_key, country_dict, prefix)

    return ord_result