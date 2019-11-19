from config.config import Config
from pca.pipeline import pca_pipeline
from config.file_system import get_path

path = get_path()

path_in = path
path_out = path

time = 'T0'
target_keys = ['country', 'status']

config = Config(path_in, path_out)

common_subjects = config.get_common_subjects()

config.path_out = path_out + '/pca/otu_counts'
otu_res = pca_pipeline(
    config,
    common_subjects,
    'otu_counts',
    'T0',
    target_keys
)

config.path_out = path_out + '/pca/nutrition'
nut_res = pca_pipeline(
    config,
    common_subjects,
    'nutrition',
    'T0',
    target_keys
)

config.path_out = path_out + '/pca/food_groups'
food_res = pca_pipeline(
    config,
    common_subjects,
    'food_groups',
    'T0',
    target_keys
)
