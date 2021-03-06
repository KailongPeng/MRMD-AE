# coding=utf-8
"""
    子集时间点Subset timepoints，用于测试被试内的流形扩展
    在测试的数据片slice上插值interpolate流形嵌入manifold embedding
    MRMD-AE
"""
"""
    代码逻辑：
        首先加载需要用到的库
        然后处理输入的超参数，放进 args 中。
        在main中

"""
# Subset timepoints for test manifold extension within subjects
# interpolate manifold embedding on test slices of data
# MRMD-AE
import os
import sys

if 'watts' in os.getcwd():
    projectDir = "/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
elif 'kailong' in os.getcwd():
    projectDir = "/Users/kailong/Desktop/MRMD-AE/MRMD-AE/"
elif 'milgram' in os.getcwd():
    projectDir = "/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/"
else:
    raise Exception('path error')
os.chdir(projectDir)
sys.path.append(f"{projectDir}/PHATE/Python/")
sys.path.append(projectDir)

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import warnings  # Ignore sklearn future warning
import numpy as np
import pandas as pd
import argparse
import torch
import random
import time
from lib.fMRI_kp import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
from lib.helper import extract_hidden_reps, get_models, checkexist, drive_decoding_kp
from torch.utils.data import DataLoader
from lib.utils import set_grad_req
from glob import glob
import phate

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ROI', type=str, default='early_visual')
parser.add_argument('--n_subjects', type=int, default=33)
parser.add_argument('--patient', type=int, default=None)
parser.add_argument('--n_timerange', type=int, default=1976)
parser.add_argument('--train_percent', type=str, default='4run')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--zdim', type=int, default=20)
parser.add_argument('--input_size', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--symm',
                    action='store_true')  # 经过查找，发现args.symm没有被使用过。编码器和解码器使用对称配置，因此，潜伏的编码器尺寸与流形尺寸相同。 use the symmetric config for encoder as decoder, so the latent encoder dim is the same as manifold dim
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--lam_mani', type=float, default=1)
parser.add_argument('--n_epochs', type=int, default=1)  # 4000

parser.add_argument('--shuffle_reg', action='store_true')
parser.add_argument('--ind_mrAE', action='store_true',
                    help='set active to train independent MR-AE')  # 设置为激活，以训练独立的MR-AE
parser.add_argument('--consecutive_time', action='store_true',
                    help='set active to make consecutive times e.g. 50% train will be first half of time series')  # 设置活动，使连续的时间，例如，50%的火车将是时间序列的前半部分。
parser.add_argument('--oneAE', action='store_true', help='use a single autoencoder')  # 使用单一的自动编码器
parser.add_argument('--reg_ref', action='store_true')

import pickle5 as pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, f)


def load_obj(name):
    if name[-3:] == 'pkl':
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


def normalize(X):
    from scipy.stats import zscore
    _X = X.copy()
    _X = zscore(_X, axis=0)
    # _X[np.isnan(_X)] = 0
    return _X


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def kp_run(cmd):
    print(cmd)
    import subprocess
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)


args = parser.parse_args()
print(f"args={args}")
subs = []
for ii in range(1, 34):
    subs.append("sub" + f"{ii}".zfill(3))
runDict = {}
for sub in subs:
    runDict[sub] = list(np.arange(1, 1 + len(glob(f"./data/localize/brain/{args.ROI}/{sub}_brain_run?.npy"))))


# runDict = {'sub001': [1, 2, 3, 4, 5, 6, 7, 8], 'sub002': [1, 2, 3, 4, 5, 6, 7], 'sub003': [1, 2, 3, 4, 5, 6, 7],
#          'sub004': [1, 2, 3, 4, 5, 6, 7, 8], 'sub005': [1, 2, 3, 4, 5, 6, 7, 8], 'sub006': [1, 2, 3, 4, 5],
#          'sub007': [1, 2, 3, 4, 5, 6, 7, 8], 'sub008': [1, 2, 3, 4, 5, 6, 7, 8], 'sub009': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub010': [1, 2, 3, 4, 5, 6, 7, 8], 'sub011': [1, 2, 3, 4, 5, 6], 'sub012': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub013': [1, 2, 3, 4, 5, 6, 7, 8], 'sub014': [1, 2, 3, 4, 5, 6, 7, 8], 'sub015': [1, 2, 3, 4, 5, 6],
#          'sub016': [1, 2, 3, 4, 5, 6, 7, 8], 'sub017': [1, 2, 3, 4, 5, 6, 7, 8], 'sub018': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub019': [1, 2, 3, 4, 5, 6, 7, 8], 'sub020': [1, 2, 3, 4, 5, 6, 7], 'sub021': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub022': [1, 2, 3, 4, 5, 6, 7, 8], 'sub023': [1, 2, 3, 4, 5, 6, 7, 8], 'sub024': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub025': [1, 2, 3, 4, 5, 6], 'sub026': [1, 2, 3, 4, 5], 'sub027': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub028': [1, 2, 3, 4, 5, 6, 7, 8], 'sub029': [1, 2, 3, 4, 5, 6, 7, 8], 'sub030': [1, 2, 3, 4, 5, 6, 7, 8],
#          'sub031': [1, 2, 3, 4, 5, 6, 7, 8], 'sub032': [1, 2, 3, 4, 5, 6, 7, 8], 'sub033': [1, 2, 3, 4, 5, 6, 7]}

def removeNanTR(brain_t, behav_t, runID):
    # 因为在数据处理的时候为了保持每一个图片都被展示5次，因此在不足5次的时候采用了Nan的补足的方法，这最初是为了方便后面的被试之间的对齐损失的设计。
    # 但是现在不想考虑那么多，就直接使用本函数去掉开头是Nan的TR
    TRhead = brain_t[:, 0]
    NanID = np.isnan(TRhead)
    brain_t = brain_t[~NanID, :]
    behav_t = behav_t[~NanID]
    runID = runID[~NanID]
    return brain_t, behav_t, runID


def trainTestSplit(subList=None, TrainingSetRun=None, presentedOnly=True):
    if TrainingSetRun is None:
        TrainingSetRun = [1, 2, 3, 4]
    if subList is None:
        subList = ['sub001', 'sub002', 'sub003']
    localizeData = './data/localize/'
    for sub in subList:
        TestingSetRun = list(set(runDict[sub]).difference(set(TrainingSetRun)))
        TestingSetRun.sort()
        print(f"{sub} TestingSetRun={TestingSetRun}")
        TrainingSet_RunNumber = len(TrainingSetRun)
        TestingSet_RunNumber = len(TestingSetRun)
        assert np.mean(np.unique((runDict[sub])) == np.unique(TrainingSetRun + TestingSetRun)) == 1
        brain = []
        behav = []
        runID = []
        if not os.path.exists(f"{localizeData}/trainTestData/{args.ROI}/{sub}_train_{TrainingSet_RunNumber}run.pkl"):
            for run in TrainingSetRun:
                brain_t = np.load(f"{localizeData}/brain/{args.ROI}/{sub}_brain_run{run}.npy")
                behav_t = np.load(f"{localizeData}/behav/{sub}_behav_run{run}.npy")
                runID_t = np.asarray([run] * len(behav_t))
                brain_t, behav_t, runID_t = removeNanTR(brain_t, behav_t, runID_t)
                brain_t = normalize(brain_t)
                if presentedOnly:
                    presentedOnly_ID = behav_t != 0
                    brain_t = brain_t[presentedOnly_ID, :]
                    behav_t = behav_t[presentedOnly_ID]
                    runID_t = runID_t[presentedOnly_ID]
                brain = brain_t if len(brain) == 0 else np.concatenate([brain, brain_t], axis=0)
                behav = behav_t if len(behav) == 0 else np.concatenate([behav, behav_t], axis=0)
                runID = runID_t if len(runID) == 0 else np.concatenate([runID, runID_t], axis=0)
            behavior = pd.DataFrame()
            behavior['behav'] = behav
            behavior['runID'] = runID
            save_obj([brain, behavior],
                     f"{localizeData}/trainTestData/{args.ROI}/{sub}_train_{TrainingSet_RunNumber}run")

            brain = []
            behav = []
            runID = []
            for run in TestingSetRun:
                brain_t = np.load(f"{localizeData}/brain/{args.ROI}/{sub}_brain_run{run}.npy")
                behav_t = np.load(f"{localizeData}/behav/{sub}_behav_run{run}.npy")
                runID_t = np.asarray([run] * len(behav_t))
                brain_t, behav_t, runID_t = removeNanTR(brain_t, behav_t, runID_t)
                brain_t = normalize(brain_t)
                if presentedOnly:
                    presentedOnly_ID = behav_t != 0
                    brain_t = brain_t[presentedOnly_ID, :]
                    behav_t = behav_t[presentedOnly_ID]
                    runID_t = runID_t[presentedOnly_ID]
                brain = brain_t if len(brain) == 0 else np.concatenate([brain, brain_t], axis=0)
                behav = behav_t if len(behav) == 0 else np.concatenate([behav, behav_t], axis=0)
                runID = runID_t if len(runID) == 0 else np.concatenate([runID, runID_t], axis=0)
            behavior = pd.DataFrame()
            behavior['behav'] = behav
            behavior['runID'] = runID
            save_obj([brain, behavior], f"{localizeData}/trainTestData/{args.ROI}/{sub}_test_{TestingSet_RunNumber}run")


trainTestSplit(subList=subs, TrainingSetRun=[1, 2, 3, 4], presentedOnly=True)


def main():
    # 首先设置重复种子，确保可重复性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # 结果应该保存在outfile
    mkdir("./results/localize/")
    outfile = './results/localize/mrmdAE_insubject_mani_extension.csv'  # outfile = 'results/mrmdAE_insubject_mani_extension.csv'

    if args.ind_mrAE:  # 如果使用独立的mr-AE。 if independent mr-AE is used
        outfile = 'results/localize/ind_mrAE_insubject_mani_extension.csv'
        if args.patient is None:
            print(
                'ERROR: need to specify subject when indepent mrAE is trained, set --pt=?')  # 错误：当独立的mrAE被训练时需要指定被试，设置--pt=?
            return
        args.n_subjects = 1
    if args.oneAE:  # 使用一个编码器一个解码器的设置
        print('using one encoder one decoder setup')
        outfile = 'results/localize/oneAE_insubject_mani_extension.csv'

    # 训练的TR的位置，也就是TR的ID ， 加载为 trainTRs
    # path_trainTRs = f"./data/mani_extension/data/sherlock_{args.train_percent}_trainTRs.npy"  # path_trainTRs = f"./data/mani_extension/data/sherlock_{args.train_percent}_trainTRs.npy"
    # if args.consecutive_time:  # 是否强调时间的连续性。如果不强调，那就是随机打乱TR
    #     path_trainTRs = f"./data/mani_extension/data/sherlock_{args.train_percent}_consec_trainTRs.npy"
    # if not os.path.exists(path_trainTRs):
    #     if not args.consecutive_time:
    #         trainTRs = np.random.choice(args.n_timerange, int(args.n_timerange * args.train_percent / 100),
    #                                     replace=False)
    #     else:
    #         trainTRs = np.arange(int(args.n_timerange * args.train_percent / 100))
    #     trainTRs.sort()
    #     np.save(path_trainTRs, trainTRs)
    # else:
    #     trainTRs = np.load(path_trainTRs)
    # # 把除了训练用的TR之外的所有TR都用作测试
    # testTRs = np.setxor1d(np.arange(args.n_timerange), trainTRs)  # numpy.setxor1d()函数查找两个数组的集合排他性，并返回输入数组中只有一个（不是两个）的排序的唯一值。
    # testTRs.sort()

    embedpath = "./data/localize/mani_extension/data/"
    if not os.path.exists(embedpath):
        os.makedirs(embedpath)

    trainTestDataPath = "./data/localize/trainTestData/"
    # datapath = "./data/localize/brain/early_visual/"  # datapath = f"data/ROI_data/{args.ROI}/fMRI"
    # datanaming = f"{args.ROI}_sherlock_movie.npy"
    embednaming = f"{args.ROI}_{args.zdim}dimension_train_{args.train_percent}_PHATE"

    # if args.consecutive_time:
    #     embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_consec_train_PHATE.npy"

    if not os.path.exists(os.path.join(embedpath, f"sub001_{embednaming}.pkl")):
        print('prepare train embed data')  # 使用 phate 准备训练嵌入数据
        for sub in subs:  # for pt in range(1, args.n_subjects + 1):
            print(sub)
            # 加载训练数据dataloader

            [X, label] = load_obj(f"{trainTestDataPath}/{args.ROI}/{sub}_train_{args.train_percent}")
            # X = np.load(os.path.join(datapath, f"sub-{pt:02}_{datanaming}"))[trainTRs]

            # 在训练数据上面训练phate模型为pop，获得训练数据X的特征值X_p
            pop = phate.PHATE(n_components=args.zdim)
            X_p = pop.fit_transform(X)

            # 加载测试数据
            testpkl = glob(f"{trainTestDataPath}/{args.ROI}/{sub}_test_?run.pkl")
            assert len(testpkl) == 1;
            testpkl = testpkl[0]
            testRunNumber = int(testpkl.split('test_')[-1].split('run.pkl')[0])
            embednaming_test = f"{args.ROI}_{args.zdim}dimension_test_{testRunNumber}run_PHATE"
            [Xtest, label_test] = load_obj(testpkl)

            # 使用已经训练好的phate模型pop来将测试数据转化到表征空间中
            Xtest_p = pop.transform(Xtest)

            # 保存训练数据的表征
            save_obj(X_p, os.path.join(embedpath, f"{sub}_{embednaming}"))

            # 保存测试数据的表征
            save_obj(Xtest_p,
                     os.path.join(embedpath,
                                  f"{sub}_{embednaming_test}"))  # 测试是phate 地标插值 Xtest_p 。 The test is phate landmark interpolation Xtest_p

    savepath = f"./data/localize/mani_extension/models/MNI152_2mm_data_{args.ROI}_mani_extend_{args.train_percent}"  # MNI152_T1_2mm_brain
    # if args.consecutive_time:
    #     savepath = savepath + '_consec'
    if args.oneAE:
        savepath = savepath + '_oneAE'

    outdf = None
    cols = ['ROI', 'hidden_dim', 'zdim', 'lam_mani', 'lam_common', 'symm_design', 'train_percent']
    entry = [args.ROI, args.hidden_dim, args.zdim, args.lam_mani, args.lam, args.symm, args.train_percent]

    if args.consecutive_time:
        entry = [args.ROI, args.hidden_dim, args.zdim,
                 args.lam_mani, args.lam,
                 args.symm, f"{args.train_percent}_consec"]

    if args.ind_mrAE:  # 培训单独的 mr-AE
        print('training individual mr-AE')
        cols.append('subject')
        entry.append(args.patient)

    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
        exist = checkexist(outdf_old, dict(zip(cols, entry)))
        if exist:
            print(f"{entry} exists")
            # return
        else:
            print(f"{entry} running")
    else:
        outdf_old = None

    patient_ids = np.arange(1, args.n_subjects + 1)
    if args.ind_mrAE:
        patient_ids = [args.patient]
    datapath = f'./data/localize/trainTestData/{args.ROI}/'
    trainTRs = args.train_percent
    datanaming = f"{args.ROI}_localize"
    # 加载训练时间点并训练 自动编码器 load training timepoints and train autoencoder
    dataset = fMRI_Time_Subjs_Embed_Dataset(patient_ids,
                                            datapath,
                                            embedpath,
                                            trainTRs,
                                            emb_name_suffix=embednaming,
                                            data_3d=False,
                                            data_name_suffix=datanaming)
    if args.input_size is None:
        args.input_size = dataset.get_TR_dims()[0]  # dataset.timeseries.shape = (320, 33, 3215)
    if args.input_size != dataset.get_TR_dims()[0]:
        print('ERROR: input dim and args.input_size not match')  # 错误：输入dim和args.input_size不匹配
        return
    if args.zdim is None:
        args.zdim = dataset.get_embed_dims()
    if args.zdim != dataset.get_embed_dims():
        print('ERROR: manifold layer dim and embedding reg dim not match')  # 错误：流形层dim和嵌入reg dim不匹配
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoders = get_models(args)
    encoder.to(device)

    if args.oneAE:
        decoders = [
            decoders[0]]  # 如果使用一个编码器一个解码器的设置，只保留一个解码器。 if use one encoder one decoder setup, keep only one decoder

    for i in range(len(decoders)):
        decoders[i].to(device)
    params = list(encoder.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)  # 要么设置初始lr，要么默认0.001。 either set initial lr or default 0.001

    criterion = torch.nn.MSELoss()  # 重建损失标准 reconstruction loss criterion
    mr_criterion = torch.nn.MSELoss()  # 流形嵌入正则化标准 manifold embedding regularization criterion
    reg_criterion = torch.nn.MSELoss()  # 惩罚不同被试的潜伏空间的错位。也就是不同被试之间的批量效应。增大lambda可以对齐不同被试之间的流形。

    losses = np.array([])
    rconst_losses = np.array([])
    reg_losses = np.array([])
    manifold_reg_losses = np.array([])

    pt_list = np.arange(len(patient_ids))

    EpochStartingTime = time.time()
    for epoch in range(1, args.n_epochs + 1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        epoch_reg_loss = 0.0

        for data_batch, embed_batch, behav_batch in dataloader:
            optimizer.zero_grad()
            current_bs = data_batch.size()[0]
            data_batch = data_batch.reshape((data_batch.shape[0] * data_batch.shape[1], -1)).float()
            data_batch = data_batch.to(device)
            embed_batch = embed_batch.reshape((embed_batch.shape[0] * embed_batch.shape[1], -1)).float()
            embed_batch = embed_batch.to(device)

            hidden = encoder(data_batch)
            hiddens = [hidden[i * current_bs:(i + 1) * current_bs] for i in range(len(patient_ids))]
            outputs = []
            embeds = []
            for i in range(len(patient_ids)):
                if args.oneAE:
                    embed, output = decoders[0](hiddens[i])
                else:
                    set_grad_req(decoders,
                                 i)  # 为了model_list的每一个模型model_list[idx]的参数parameters都设置一个梯度的要求param.requires_grad = True  set gradient requirement
                    embed, output = decoders[i](hiddens[i])
                outputs.append(output)
                embeds.append(embed)

            # if args.shuffle_reg:
            #     random.shuffle(pt_list)

            if args.lam > 0:  # 惩罚不同被试的潜伏空间的错位。也就是不同被试之间的批量效应。增大lambda可以对齐不同被试之间的流形。

                # 确保当前的batch的数据的所有的TR分别对应的图片都是一致的。
                def sameTRlabel(t1, t2):
                    t1 = np.asarray(t1)
                    t2 = np.asarray(t2)
                    assert np.mean(t1 == t2) == 1

                sameTRlabel(behav_batch[:, pt_list[0]], behav_batch[:, pt_list[1]])  # behav_batch: TR_subset x sub

                loss_reg = reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[1]])
                if args.reg_ref:
                    for z1 in range(1, len(patient_ids)):
                        loss_reg += reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[z1]])
                else:
                    for z1 in range(1, len(patient_ids) - 1):  # 比较 所有相邻对的被试的隐藏层的MSE差别。 consecutive pairs (cycle)
                        z2 = z1 + 1
                        loss_reg += reg_criterion(hiddens[pt_list[z1]], hiddens[pt_list[z2]])

            loss_reconstruct = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)  # 比较原始数据和重建的数据的差别
            loss_manifold_reg = mr_criterion(torch.stack(embeds).view(embed_batch.shape), embed_batch)

            loss = loss_reconstruct + args.lam_mani * loss_manifold_reg  # 这是真正用来训练网络权重的loss，包括重建算是和流形损失，在下一行还可能包括被试对齐损失

            if args.lam > 0:  # 不同被试之间的对齐
                loss += args.lam * loss_reg

            loss.backward()
            optimizer.step()

            # 保存训练过程中得到的各种loss， 包括 epoch_losses所有loss的和 ；epoch_rconst_losses ； epoch_manifold_reg_losses ； epoch_reg_loss
            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item() * data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)

            if args.lam > 0:
                epoch_reg_loss += loss_reg.item() * data_batch.size(0)

        epoch_losses = epoch_losses / (len(trainTRs) * len(patient_ids))  # 改为报告历时损失。 change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / (len(trainTRs) * len(patient_ids))
        epoch_manifold_reg_losses = epoch_manifold_reg_losses / (len(trainTRs) * len(patient_ids))
        epoch_reg_loss = epoch_reg_loss / (len(trainTRs) * len(patient_ids))

        CurrentTime = time.time()

        print(
            f"Epoch {epoch}\tLoss={epoch_losses:.4f}\tloss_rconst={epoch_rconst_losses:.4f}\tloss_manfold_reg={epoch_manifold_reg_losses:.4f}\tloss_reg={epoch_reg_loss:.4f}\ttime passed={int((CurrentTime - EpochStartingTime)/60)}min")

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)
        reg_losses = np.append(reg_losses, epoch_reg_loss)

        # 在训练过程中就开始保存 loss
        all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, reg_losses), axis=1)
        lossfile = f'mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_all_train_losses.npy'
        np.save(os.path.join(savepath, lossfile), all_losses)

    all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, reg_losses), axis=1)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    lossfile = f'mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_all_train_losses.npy'
    # if args.ind_mrAE:
    #     lossfile = f'ind_mrAE_sub-{args.patient:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_all_train_losses.npy'
    np.save(os.path.join(savepath, lossfile), all_losses)

    modeldict = {'encoder_state_dict': encoder.state_dict()}
    for i in range(len(decoders)):
        modeldict[f"decoder_{i}_state_dict"] = decoders[i].state_dict()
    modeldict['optimizer_state_dict'] = optimizer.state_dict()
    modeldict['epoch'] = epoch

    # 保存checkpoint文件，防止模型半途崩溃。
    ckptfile = f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}.pt"
    if args.ind_mrAE:
        ckptfile = f'ind_mrAE_sub-{args.patient:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}.pt'
    torch.save(modeldict, os.path.join(savepath, ckptfile))
    testTRs = 'test'
    # 在测试时间点上进行测试，并记录测试嵌入。 test on test timepoints and record the test embeddings
    dataset = fMRIAutoencoderDataset(patient_ids,
                                     datapath,
                                     testTRs,
                                     data_3d=False,
                                     data_name_suffix=datanaming)
    encoder.eval()
    hidden, al_hidden, behav_test, runID_test = extract_hidden_reps(encoder, decoders, dataset, device, None, args)
    # hidden = hidden.reshape(args.n_subjects, len(testTRs), -1)

    if args.ind_mrAE:
        hidden = hidden.reshape(len(testTRs), -1)
        hiddenfile = f"ind_mrAE_sub-{args.patient:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden"
    else:
        hiddenfile = f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden"
    save_obj([hidden, behav_test, runID_test], os.path.join(savepath, hiddenfile))
    # np.save(os.path.join(savepath, hiddenfile), hidden)

    cols.append('hiddenfile')
    entry.append(os.path.join(savepath, hiddenfile))

    if outdf is None:
        outdf = pd.DataFrame(columns=cols)
    outdf.loc[len(outdf)] = entry

    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
        outdf = pd.concat([outdf_old, outdf])
    outdf.to_csv(outfile, index=False)


def get_decoding(behav_test, runID_test, args, embeds=None):
    entry = []
    test_embeds = []
    if embeds is not None:
        test_embeds = embeds
    else:
        pass

    labels = behav_test
    runID = runID_test
    results = drive_decoding_kp(test_embeds, labels, runID, balance_min=False)

    entry.append(results['accuracy'].mean())
    entry.append(results['accuracy'].std())
    return entry, results


def testDataClassification(args):
    savepath = f"./data/localize/mani_extension/models/MNI152_2mm_data_{args.ROI}_mani_extend_{args.train_percent}"  # MNI152_T1_2mm_brain
    hiddenfile = f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden"
    [hidden, behav_test, runID_test] = load_obj(os.path.join(savepath, hiddenfile))

    cols = ['train_percent', 'ROI', 'method', 'mean', 'std']
    outdf = pd.DataFrame(columns=cols)
    print("get MR-AE decodings ...")  # 获得MR-AE解码
    entry = ['4run', args.ROI, 'MRMD-AE']
    entry_add, results = get_decoding(behav_test, runID_test, args, embeds=hidden)
    entry.extend(entry_add)
    outdf.loc[len(outdf)] = entry

    print(outdf)

    # 保存结果
    save_obj([entry_add, results], f"{savepath}/testDataClassificationResult")

    return entry, results

def checkLoss():
    # http://localhost:8288/lab/workspaces/auto-8/tree/users/kp578/localize/MRMD-AE/archive/testMRMD-AE.ipynb
    # /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/data/localize/mani_extension/models/MNI152_2mm_data_early_visual_mani_extend_4run/
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle5 as pickle
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(obj, f)

    def load_obj(name):
        if name[-3:] == 'pkl':
            with open(name, 'rb') as f:
                return pickle.load(f)
        else:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)
    os.chdir("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
    ROI = 'early_visual'
    train_percent = '4run'
    hidden_dim = 64
    zdim = 20
    lam = float(100)
    lam_mani = float(100)
    symm = False
    savepath = f"./data/localize/mani_extension/models/MNI152_2mm_data_{ROI}_mani_extend_{train_percent}"
    lossfile = f'mrmdAE_{hidden_dim}_{zdim}_lam{lam}_manilam{lam_mani}_symm{symm}_all_train_losses.npy'
    all_losses = np.load(os.path.join(savepath, lossfile))
    for whichLoss, currLoss in enumerate(['3losses', 'rconst_losses', 'manifold_reg_losses', 'reg_losses']):
        lossRecord = all_losses[:, whichLoss]
        _ = plt.figure()
        _ = plt.plot(lossRecord)
        _ = plt.ylim([0, lossRecord[0] + 10])
        _ = plt.title(currLoss)
    [entry_add, results] = load_obj(f"{savepath}/testDataClassificationResult")

if __name__ == '__main__':
    main()
    entry, results = testDataClassification(args)
    print('done')
