import os
import warnings  # Ignore sklearn future warning
import numpy as np
import pandas as pd
import argparse
import torch
import random
from glob import glob
import phate
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
from lib.helper import extract_hidden_reps, get_models, checkexist
from torch.utils.data import DataLoader
from lib.utils import set_grad_req
warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default=0)
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--n_subjects', type = int, default=16)


def main():
    args = parser.parse_args()

    # 首先设置重复种子，确保可重复性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/data/"
    subs = glob(f"{subFolder}/sub*")

    """
        localize的原始数据
        anat数据
            /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/anat
        fMRI的数据
            /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/func
        行为学的数据
            包括是否包含黑白斑块，是否按键了，当前的图片是哪一张
            /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/behav/
        眼动记录仪的数据
            暂时不知道
    """
    # 首先得到所有的被试的fMRI的数据以及对应的行为学数据
    # 把所有的fMRI的数据转移到标准空间中去
    # 获得ROI的mask
    # 最终的结果就是ROI的所有的标准空间中的数据加上行为学的数据


if __name__=='__main__':
    main()
