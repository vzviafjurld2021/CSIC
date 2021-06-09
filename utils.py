import os,sys
import numpy as np
import random
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from torch._six import inf
import pandas as pd
from PIL import Image
from sklearn.feature_extraction import image
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import *
from arguments import get_args
args = get_args()

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


########################################################################################################################

class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))