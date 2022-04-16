testMode = True
# http://localhost:8970/notebooks/users/kp578/localize/MRMD-AE/archive/dataPreparation.ipynb
import os
import warnings  # Ignore sklearn future warning
import numpy as np
import pandas as pd
import argparse
# import torch
import random
from glob import glob
import subprocess
import nibabel as nib
from tqdm import tqdm
import sys
import pickle5 as pickle
import time

sys.path.append("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
os.chdir("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
# import phate
# from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
# from lib.helper import extract_hidden_reps, get_models, checkexist
# from torch.utils.data import DataLoader
# from lib.utils import set_grad_req
warnings.simplefilter(action='ignore', category=FutureWarning)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response or 'Unrecognised' in sbatch_response:
        raise Exception(sbatch_response)


def getjobID_num(sbatch_response):  # 根据subprocess.Popen输出的proc，获得sbatch的jpobID
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID


def kp_run(cmd):
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def kp_remove(fileName):
    cmd = f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)


def checkEndwithDone(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    return last_line == "done\n"


def checkDone(jobIDs):
    completed = {}
    for jobID in jobIDs:
        filename = f"./logs/{jobID}.out"
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values())) == 1:
        status = True
    else:
        status = False
    return completed, status


def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status == True:
        pass
    else:
        print(completed)
        assert status == True
    return completed


def check_jobArray(jobID='', jobarrayNumber=10):
    arrayIDrange = np.arange(1, 1 + jobarrayNumber)
    jobIDs = []
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed


def waitForEnd(jobID):
    while jobID_running_myjobs(jobID):
        print(f"waiting for {jobID} to end")
        time.sleep(5)
    print(f"{jobID} finished")


def jobID_running_myjobs(jobID):
    jobID = str(jobID)
    cmd = "squeue -u kp578"
    sbatch_response = subprocess.getoutput(cmd)
    if jobID in sbatch_response:
        return True
    else:
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ROI', type=str, default='early_visual')
parser.add_argument('--n_subjects', type=int, default=16)


def main():
    # args = parser.parse_args()
    args = parser.parse_args("")

    # 首先设置重复种子，确保可重复性
    np.random.seed(args.seed)
    random.seed(args.seed)

    subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
    subs = glob(f"{subFolder}/sub*");
    subs.sort();
    subs = [sub.split("/")[-1] for sub in subs]
    for sub in subs:
        funcs = glob(f"{subFolder}/{sub}/func/*.nii")
        print(f"{sub} {len(funcs)}")

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

    # 首先把所有的functional的数据全部转移到functional template的空间中
    def alignFunc(sub='', arrayJobs={}):
        transformFolder = f"{subFolder}/{sub}/transform/"
        mkdir(transformFolder)

        # 首先把所有的functional的数据全部都进行运动校正，然后选出一个模板。
        def saveMiddleVolumeAsTemplate(sub='', funcFolder='',
                                       transformFolder=''):  # /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/func/sub022_func01.nii
            func1head = f'{sub}_func01'
            nii = nib.load(f"{funcFolder}/{func1head}.nii")
            frame = nii.get_fdata()
            TR_number = frame.shape[3]
            frame = frame[:, :, :, int(TR_number / 2)]
            frame = nib.Nifti1Image(frame, affine=nii.affine)
            template = f"{transformFolder}/{func1head}_template"
            nib.save(frame, template)
            return template

        funcTemplate = saveMiddleVolumeAsTemplate(sub=sub, funcFolder=f"{subFolder}/{sub}/func/",
                                                  transformFolder=transformFolder)  # 把 第一个 run 的中间的volume作为 模板

        funcs = glob(f"{subFolder}/{sub}/func/*.nii")
        funcs.sort()
        for func in funcs:  # func = '/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects//sub022/func/sub022_func01.nii'
            currJobID = len(arrayJobs) + 1
            func_id = int(func.split('func')[-1].split('.')[0])  # ii + 1
            arrayJobs[currJobID] = [func_id, func, transformFolder, funcTemplate]

            def convert_func_to_templateSpace(func_id, func, transformFolder, funcTemplate):
                funcHead = func.split('/')[-1].split('.')[0]  # funcHead = 'sub022_func01'
                mcFile = f"{transformFolder}/{funcHead}_mc.nii.gz"
                if not testMode:
                    kp_remove(mcFile)
                mcFile_template = f"{transformFolder}/{funcHead}_mc_template.nii.gz"
                cmd = f"mcflirt -in {func} -out {mcFile}";
                kp_run(cmd);
                wait(mcFile)
                func2temp = f"{transformFolder}/run{func_id}_to_template.mat"
                cmd = f"flirt -in {mcFile} -out {transformFolder}/{funcHead}_mc_temporary.nii.gz -ref {funcTemplate} -dof 6 -omat {func2temp}";
                kp_run(cmd);
                wait(func2temp)
                cmd = f"flirt -in {mcFile} -out {mcFile_template} -ref {funcTemplate} -applyxfm -init {transformFolder}/run{func_id}_to_template.mat";
                kp_run(cmd);
                wait(mcFile_template)
                kp_remove(f"{transformFolder}/{funcHead}_mc_temporary.nii.gz")
                print("done")
            # convert_func_to_templateSpace(func_id, func, transformFolder, funcTemplate)
        return arrayJobs

    arrayJobs = {}
    for sub in subs:
        arrayJobs = alignFunc(sub=sub, arrayJobs=arrayJobs)
    save_obj(arrayJobs,
             f"/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/convert_func_to_templateSpace")
    cmd = f"sbatch --requeue --array=1-{len(arrayJobs)} /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/convert_func_to_templateSpace.sh ";
    sbatch_response = kp_run(cmd);
    jobID = getjobID_num(sbatch_response);
    waitForEnd(jobID);
    check_jobArray(jobID=jobID, jobarrayNumber=len(arrayJobs))

    # 把所有的functional的数据转移到标准空间中并且应用mask
    jobIDs = {}
    jobID = 1
    for sub in tqdm(subs):
        jobIDs[jobID] = sub
        jobID += 1
    save_obj(jobIDs,
             f"/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/transformSubjectDataIntoStand")
    cmd = f"sbatch --requeue --array 1-{len(jobIDs)} /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/transformSubjectDataIntoStand.sh"
    sbatch_response = kp_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    check_jobArray(jobID=jobID, jobarrayNumber=len(jobIDs))

    # 准备对于每一个run内部的图片进行对齐。 不是每一个被试都有8个run，不是所有的图片在一个run中都被展示了8次。
    # 同时进行BOLD lag的补偿。
    def sort_align_behav_brain():

        import string
        alphabet = string.ascii_uppercase

        def convertItemColumn(ShownImages):
            ShownImages_ = []
            for image in ShownImages:
                type(image)
                if type(image) == str:
                    imageID = alphabet.index(image) + 1
                    ShownImages_.append(imageID)
                elif type(image) == float:
                    ShownImages_.append(0)
            return np.asarray(ShownImages_)

        def loadBrainData(behavFile):
            run_ID = behavFile.split('_')[-1].split('.')[0]
            brain = np.load(
                f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/transform/{sub}_func0{run_ID}_mc_template_inStand_masked.npy")
            brain = brain.T
            print(f"brain.shape={brain.shape}")
            # 最简单的解决BOLD lag的方法就是把functional的数据的前三个TR都丢掉。
            brain = brain[3:, :]
            print(f"brain.shape={brain.shape}")
            return brain

        def complete5ID(ID):  # 对于sub019 的第五个run，第四张图片展示了4次，第七张图片展示了3次。此时需要补齐5个ID才可以。
            completeID = []
            for ii in range(5):
                try:
                    completeID.append(ID[ii])
                except:
                    completeID.append(None)
            return np.asarray(completeID)

        def reSortBasedOnImages(ShownImages):
            reSortedLabels = []
            IDs = 0
            for image in range(1, 17):
                ID = np.where(ShownImages == image)[0]
                ID = complete5ID(ID)
                IDs = ID if image == 1 else np.concatenate([IDs, ID], axis=0)
                reSortedLabels = reSortedLabels + 5 * [image]
            ID = np.where(ShownImages == 0)[0]
            IDs = np.concatenate([IDs, ID], axis=0)
            reSortedLabels = reSortedLabels + len(ID) * [0]
            return IDs, reSortedLabels

        subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
        subs = glob(f"{subFolder}/sub*")
        subs.sort()
        subs = [sub.split("/")[-1] for sub in subs]
        for sub in tqdm(subs):
            behavFolder = f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/behav/???_?.csv"
            behavFiles = glob(behavFolder)
            behavFiles.sort()
            for behavFile in behavFiles:
                runID = behavFile.split('_')[-1].split('.')[0]
                brain = loadBrainData(behavFile)
                behav = pd.read_csv(behavFile)
                ShownImages = convertItemColumn(list(behav.Item))
                print(f"brain.shape[0]={brain.shape[0]} len(ShownImages)={len(ShownImages)}")
                if brain.shape[0] < len(ShownImages):  # 一般来说是行为学的数据长于大脑数据，此时删除部分行为学数据
                    ShownImages = ShownImages[:brain.shape[0]]
                    print('行为学数据长')
                else:  # 偶尔也会行为学的数据短于大脑数据，此时删除部分大脑数据。
                    brain = brain[:len(ShownImages)]

                assert len(ShownImages) == brain.shape[0]
                IDs, reSortedLabels = reSortBasedOnImages(ShownImages)
                print(f"IDs={IDs}")
                print(f"reSortedLabels={reSortedLabels}")
                if np.sum((1 * (IDs == None))) == 0:
                    IDs = IDs.astype(int)
                    brain = brain[IDs, :]
                else:
                    resortedBrain = []
                    for ID in IDs:
                        if ID is None:
                            temp = brain[0, :]
                            temp[:] = None
                            resortedBrain.append(temp)
                        else:
                            resortedBrain.append(brain[ID, :])
                    brain = np.asarray(resortedBrain)

                np.save(f"{subFolder}/{sub}/transform/{sub}_brain_run{runID}", brain)
                np.save(f"{subFolder}/{sub}/transform/{sub}_behav_run{runID}", reSortedLabels)

    sort_align_behav_brain()

    def scpfiles():
        import subprocess,os
        def kp_run(cmd):
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)
            return sbatch_response

        def mkdir(folder):
            if not os.path.isdir(folder):
                cmd = f"mkdir -p {folder}"
                sbatch_response = subprocess.getoutput(cmd)
                print(sbatch_response)
        "cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/ ; mkdir temp"
        "rsync -a subjects/sub0??/transform/sub0??_behav_run?.npy temp/"
        "rsync -a subjects/sub0??/transform/sub0??_brain_run?.npy temp/"
        "scp kp578@milgram.hpc.yale.edu:/gpfs/milgram/project/turk-browne/projects/localize/analysis/temp/* /Users/kailong/Desktop/MRMD-AE/MRMD-AE/data/localize"


if __name__ == '__main__':
    main()

# def transformSubjectDataIntoStand(sub=''):
#     transformFolder = f"{subFolder}/{sub}/transform/"
#     funcTemplate = f"{transformFolder}/{sub}_func01_template.nii"
#     func2anat = f"{transformFolder}/func2anat.mat"
#     anat2stand = f"{transformFolder}/anat2stand.mat"
#     func2stand = f"{transformFolder}/func2stand.mat"
#
#     func = funcTemplate
#     anat = f"{subFolder}/{sub}/anat/{sub}_t1_bet.nii.gz"
#     stand = '/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz'
#     funcInAnat = f"{transformFolder}/funcInAnat.nii.gz"
#     anatInStand = f"{transformFolder}/anatInStand.nii.gz"
#
#     # 获得func2anat
#     cmd = f"flirt -in {func} -out {funcInAnat} -ref {anat} -omat {func2anat} -dof 6" ; kp_run(cmd)
#
#     # 获得anat2stand
#     cmd = f"flirt -in {anat} -out {anatInStand} -ref {stand} -omat {anat2stand}" ; kp_run(cmd)
#
#     # 获得func2stand
#     cmd = f"convert_xfm -omat {func2stand} -concat {func2anat} {anat2stand}" ; kp_run(cmd)
#
#     # 使用func2stand
#     mask = nib.load("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/early_visual_association-test_z_FDR_0.01.nii.gz").get_fdata() ;
#     mask[mask > 0] = 1
#     print(f"mask.shape={mask.shape}") ; print(f"np.sum(mask)={np.sum(mask)}")
#     funcs = glob(f"{transformFolder}/*_mc_template.nii.gz") ; funcs.sort()   #sub022_func04.nii
#     for funcRun in funcs:
#         name = funcRun.split('/')[-1].split('.')[0]
#         print(f"{sub} {name}")
#         funcRun_inStand = f"{transformFolder}/{name}_inStand.nii.gz"
#         cmd = f"flirt -in {funcRun} -out {funcRun_inStand} -ref {stand} -applyxfm -init {func2stand}" ; kp_run(cmd)
#         # 使用mask
#         funcRun_inStand_matrix = nib.load(funcRun_inStand).get_fdata()
#         funcRun_inStand_matrix_masked = funcRun_inStand_matrix[mask>0]
#         print(f"funcRun_inStand_matrix_masked.shape={funcRun_inStand_matrix_masked.shape}")
#         funcRun_inStand_masked = f"{transformFolder}/{name}_inStand_masked"
#         np.save(funcRun_inStand_masked, funcRun_inStand_matrix_masked)
