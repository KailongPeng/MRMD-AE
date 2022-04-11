testMode = True
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


# def convert_func_to_templateSpace(func_id,func,transformFolder,funcTemplate):
#     funcHead = func.split('/')[-1].split('.')[0] # funcHead = 'sub022_func01'
#     mcFile = f"{transformFolder}/{funcHead}_mc.nii.gz"
#     if not testMode:
#         kp_remove(mcFile)
#     mcFile_template = f"{transformFolder}/{funcHead}_mc_template.nii.gz"
#     cmd = f"mcflirt -in {func} -out {mcFile}" ; kp_run(cmd) ; wait(mcFile)
#     func2temp = f"{transformFolder}/run{func_id}_to_template.mat"
#     cmd = f"flirt -in {mcFile} -out {transformFolder}/{funcHead}_mc_temporary.nii.gz -ref {funcTemplate} -dof 6 -omat {func2temp}" ; kp_run(cmd) ; wait(func2temp)
#     cmd = f"flirt -in {mcFile} -out {mcFile_template} -ref {funcTemplate} -applyxfm -init {transformFolder}/run{func_id}_to_template.mat" ; kp_run(cmd) ; wait(mcFile_template)
#     kp_remove(f"{transformFolder}/{funcHead}_mc_temporary.nii.gz")
#     print("done")
def transformSubjectDataIntoStand(sub=''):
    subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
    transformFolder = f"{subFolder}/{sub}/transform/"
    funcTemplate = f"{transformFolder}/{sub}_func01_template.nii"
    func2anat = f"{transformFolder}/func2anat.mat"
    anat2stand = f"{transformFolder}/anat2stand.mat"
    func2stand = f"{transformFolder}/func2stand.mat"

    func = funcTemplate
    anat = f"{subFolder}/{sub}/anat/{sub}_t1_bet.nii.gz"
    stand = '/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz'
    funcInAnat = f"{transformFolder}/funcInAnat.nii.gz"
    anatInStand = f"{transformFolder}/anatInStand.nii.gz"

    # 获得func2anat
    cmd = f"flirt -in {func} -out {funcInAnat} -ref {anat} -omat {func2anat} -dof 6" ; kp_run(cmd)

    # 获得anat2stand
    cmd = f"flirt -in {anat} -out {anatInStand} -ref {stand} -omat {anat2stand}" ; kp_run(cmd)

    # 获得func2stand
    cmd = f"convert_xfm -omat {func2stand} -concat {func2anat} {anat2stand}" ; kp_run(cmd)

    # 使用func2stand
    mask = nib.load("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/early_visual_association-test_z_FDR_0.01.nii.gz").get_fdata() ;
    mask[mask > 0] = 1
    print(f"mask.shape={mask.shape}") ; print(f"np.sum(mask)={np.sum(mask)}")
    funcs = glob(f"{transformFolder}/*_mc_template.nii.gz") ; funcs.sort()   #sub022_func04.nii
    # for funcRun in funcs:
    #     name = funcRun.split('/')[-1].split('.')[0]
    #     print(f"{sub} {name}")
    #     funcRun_inStand = f"{transformFolder}/{name}_inStand.nii.gz"
    #     cmd = f"flirt -in {funcRun} -out {funcRun_inStand} -ref {stand} -applyxfm -init {func2stand}" ; kp_run(cmd)
    #     # 使用mask
    #     funcRun_inStand_matrix = nib.load(funcRun_inStand).get_fdata()
    #     funcRun_inStand_matrix_masked = funcRun_inStand_matrix[mask>0]
    #     print(f"funcRun_inStand_matrix_masked.shape={funcRun_inStand_matrix_masked.shape}")
    #     funcRun_inStand_masked = f"{transformFolder}/{name}_inStand_masked"
    #     np.save(funcRun_inStand_masked, funcRun_inStand_matrix_masked)
    jobIDs={}
    jobID = 1
    for funcRun in funcs:
        jobIDs[jobID] = [sub,funcRun,transformFolder]
        jobID+=1
    save_obj(jobIDs, f"/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/temp/usefunc2stand_{sub}")
    cmd = f"sbatch --requeue --array 1-{len(jobIDs)} /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/usefunc2stand.sh {sub}"
    sbatch_response = kp_run(cmd);
    jobID = getjobID_num(sbatch_response);
    waitForEnd(jobID);
    check_jobArray(jobID=jobID, jobarrayNumber=len(arrayJobs))


    print("done")

jobID = int(float(sys.argv[1]))
arrayJobs = load_obj(f"/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/transformSubjectDataIntoStand")
sub = arrayJobs[jobID]
print(f"sub={sub}")
transformSubjectDataIntoStand(sub=sub)