from config.otu_counts import load_otu_counts
from config.nutrition import load_nutrition
from config.food_groups import load_food_groups
from config.subjects import load_subject_info, T0_T1_subject_separation
import pandas as pd
from pcoa.pipeline import pcoa_pipeline
from scipy.spatial import procrustes
import numpy as np


class Config:

    def __init__(self,
                 in_path,
                 out_path
                 ):

        self.in_path = in_path
        self.out_path = out_path

        self.subject_info = load_subject_info(self.in_path + '/' + 'correct_subject_info.tsv')
        self.subject_info_T0, self.subject_info_T1 = T0_T1_subject_separation(self.subject_info)

        self.food_groups = load_food_groups(self.in_path + '/' + 'food_groups_long.tsv', norm='normalize')

        self.nutrition = load_nutrition(self.in_path + '/' + 'nutrition_long.tsv', norm='normalize')

        self.otu_counts = load_otu_counts(self.in_path + '/' + 'OTUcounts.tsv', norm='none')

        