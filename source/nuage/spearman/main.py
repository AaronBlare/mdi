import os
from config.config import Config
from spearman.pipeline import pipeline_spearman
from config.file_system import get_path

path = get_path()

data_file_path = path
tables_file_path = path + '/spearman/table/'

if not os.path.isdir(tables_file_path):
    os.makedirs(tables_file_path)

config = Config(data_file_path, tables_file_path)

pipeline_spearman(config)
