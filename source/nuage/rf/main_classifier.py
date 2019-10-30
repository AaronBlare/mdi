from config.config import Config
from rf.pipeline_classifier import pipeline_classifier

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
figure_file_path = 'D:/Aaron/Bio/NU-Age/Figures/'

config = Config(data_file_path, figure_file_path)

pipeline_classifier(config)
