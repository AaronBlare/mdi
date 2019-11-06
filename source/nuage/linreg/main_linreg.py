from config.config import Config
from linreg.pipeline_linreg import pipeline_linreg

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'

config = Config(data_file_path, figure_file_path)

pipeline_linreg(config)
