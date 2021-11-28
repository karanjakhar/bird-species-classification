from train import training
from test import testing
import torch
from config import CONFIG
import os
import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('../data/'):
    os.mkdir('../data/')
if not os.path.exists('../model_weights'):
    os.mkdir('../model_weights')



if __name__ == '__main__':
    training()
    net = torch.load('../model_weights/last.pth').to(CONFIG['device'])
    testing(net)