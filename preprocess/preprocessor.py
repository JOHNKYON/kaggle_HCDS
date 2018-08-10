"""This module is the data preprocessor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


class Preprocessor:
    """This is a class contains functions of data preprocess"""

    def read_data(self, file_name):
        """Read data by file_name"""
        if file_name == "train":
            file = pd.read_csv("data/application_train.csv")

