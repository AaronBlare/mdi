from config.config import Config
from pcoa.pipeline import pcoa_pipeline
from infrastructure.file_system import get_path

path = get_path()

data_file_path = path
result_file_path = path
config = Config(data_file_path, result_file_path)

time = 'T0'
target_keys = ['country', 'status']

common_subjects = config.get_common_subjects_with_adherence()

config.path_out = result_file_path + '/pcoa/otu_counts'
otu_res = pcoa_pipeline(
    config,
    common_subjects,
    'otu_counts',
    'T0',
    target_keys
)

config.path_out = result_file_path + '/pcoa/nutrition'
nut_res = pcoa_pipeline(
    config,
    common_subjects,
    'nutrition',
    'T0',
    target_keys
)

config.path_out = result_file_path + '/pcoa/food_groups'
food_res = pcoa_pipeline(
    config,
    common_subjects,
    'food_groups',
    'T0',
    target_keys
)
