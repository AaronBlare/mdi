import os
from config.config import Config
from linreg.pipeline import pipeline_linreg

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
tables_file_path = 'D:/Aaron/Bio/NU-Age/Linreg/Table/'

if not os.path.isdir(tables_file_path):
    os.makedirs(tables_file_path)

config = Config(data_file_path, tables_file_path)

pipeline_linreg(config)
