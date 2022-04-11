testMode=True
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
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)
def getjobID_num(sbatch_response): # 根据subprocess.Popen输出的proc，获得sbatch的jpobID
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID
def kp_run(cmd):
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    jobID = getjobID_num(sbatch_response)
    return jobID
def kp_remove(fileName):
    cmd=f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)
def wait(tmpFile,waitFor=0.1):
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
    return last_line=="done\n"
def checkDone(jobIDs):
    completed={}
    for jobID in jobIDs:
        filename = f"./logs/{jobID}.out"
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values()))==1:
        status = True
    else:
        status = False
    return completed, status
def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status==True:
        pass
    else:
        print(completed)
        assert status==True
    return completed
def check_jobArray(jobID='',jobarrayNumber=10):
    arrayIDrange=np.arange(1,1+jobarrayNumber)
    jobIDs=[]
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed

def convert_func_to_templateSpace(func_id,func,transformFolder,funcTemplate):
    funcHead = func.split('/')[-1].split('.')[0] # funcHead = 'sub022_func01'
    mcFile = f"{transformFolder}/{funcHead}_mc.nii.gz"
    if not testMode:
        kp_remove(mcFile)
    mcFile_template = f"{transformFolder}/{funcHead}_mc_template.nii.gz"
    cmd = f"mcflirt -in {func} -out {mcFile}" ; kp_run(cmd) ; wait(mcFile)
    func2temp = f"{transformFolder}/run{func_id}_to_template.mat"
    cmd = f"flirt -in {mcFile} -out {transformFolder}/{funcHead}_mc_temporary.nii.gz -ref {funcTemplate} -dof 6 -omat {func2temp}" ; kp_run(cmd) ; wait(func2temp)
    cmd = f"flirt -in {mcFile} -out {mcFile_template} -ref {funcTemplate} -applyxfm -init {transformFolder}/run{func_id}_to_template.mat" ; kp_run(cmd) ; wait(mcFile_template)
    kp_remove(f"{transformFolder}/{funcHead}_mc_temporary.nii.gz")
    print("done")
jobID = int(float(sys.argv[1]))
arrayJobs = load_obj(f"/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/convert_func_to_templateSpace")
[func_id, func, transformFolder, funcTemplate] = arrayJobs[jobID]
print(f"func_id={func_id}, func={func}, transformFolder={transformFolder}, funcTemplate={funcTemplate}")
convert_func_to_templateSpace(func_id, func, transformFolder, funcTemplate)