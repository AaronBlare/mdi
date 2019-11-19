import os
from config.config import Config
from linreg.pipeline import pipeline_diff_linreg, pipeline_T0_T1_linreg
from config.file_system import get_path

path = get_path()

data_file_path = path
tables_file_path = path + '/linreg/table/'

if not os.path.isdir(tables_file_path):
    os.makedirs(tables_file_path)

config = Config(data_file_path, tables_file_path)

pipeline_T0_T1_linreg(config)

pipeline_diff_linreg(config)


