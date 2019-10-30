from config.config import Config
from rf.pipeline_regressor import pipeline_regressor

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'

config = Config(data_file_path, figure_file_path)

pipeline_regressor(config)
