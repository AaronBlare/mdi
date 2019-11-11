import os
from config.config import Config
from linreg.plot import pipeline_plot

data_file_path = 'D:/Aaron/Bio/NU-Age/Data/'
figures_file_path = 'D:/Aaron/Bio/NU-Age/Linreg/Figures/'

otu_file = 'otus.txt'
plot_type = 'subject'  # 'subject' for subject-control separation, 'all' - without separation

if not os.path.isdir(figures_file_path):
    os.makedirs(figures_file_path)

config = Config(data_file_path, figures_file_path)

pipeline_plot(config, otu_file, plot_type)
