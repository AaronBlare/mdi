from config.config import Config
from linreg.pipeline import pipeline_linreg

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
tables_file_path = 'D:/Aaron/Bio/NU-Age/Linreg/'

config = Config(data_file_path, tables_file_path)

pipeline_linreg(config)
