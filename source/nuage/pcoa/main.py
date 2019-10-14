from config.config import Config
from pcoa.pipeline import pcoa_pipeline

path_in = 'D:/YandexDisk/Work/nuage'
path_out = 'D:/YandexDisk/Work/nuage'

time = 'T0'
target_keys = ['country', 'status']

config = Config(path_in, path_out)

common_subjects = config.get_common_subjects()

config.path_out = path_out + '/otu_counts'
otu_res = pcoa_pipeline(
    config,
    common_subjects,
    'otu_counts',
    'T0',
    target_keys
)

config.path_out = path_out + '/nutrition'
nut_res = pcoa_pipeline(
    config,
    common_subjects,
    'nutrition',
    'T0',
    target_keys
)

config.path_out = path_out + '/food_groups'
food_res = pcoa_pipeline(
    config,
    common_subjects,
    'food_groups',
    'T0',
    target_keys
)
