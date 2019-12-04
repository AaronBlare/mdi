import os
from config.config import Config
from rf.pipeline_classifier import *
from infrastructure.file_system import get_path

path = get_path()

data_file_path = path
result_file_path = path + '/rf_classifier/countries'

if not os.path.isdir(result_file_path):
    os.makedirs(result_file_path)

config = Config(data_file_path, result_file_path)

pipeline_classifier_countries(config)
