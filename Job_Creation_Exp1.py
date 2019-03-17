import numpy as np
import pandas as pd
import json
import sys
from shutil import copyfile
from pprint import pprint
import os

#################################################################
#################################################################

# Full Path of where the experiment results are to be stored.
ExpsPath = "/home/aris/Desktop/Measurements/"

# Default JSON file
defaultJsonPath = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Default_DLIB_inputs_Exp1.json"

# Default JSON file
defaultPBSPath = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Default_Job_Exp1.pbs"

# Thresholds for the experiments
thresholds = np.array([0.0008, 0.008, 0.08])

# Batch Sizes for the experiments
batchSizes = np.array([64, 128, 192])

# Number of sites
sites = np.array([4, 8, 16, 32, 64])

# Number of actual jobs
jobs = np.shape(thresholds)[0]*np.shape(batchSizes)[0]*np.shape(sites)[0]

mesh = np.array(np.meshgrid(thresholds, batchSizes, sites)).T.reshape(jobs,3)
# print(jobs)
# print(mesh)
# print(np.shape(mesh))

#################################################################
#################################################################

### Creating the JSON and PBS files for all the jobs. ###

for i in range(jobs):
    
    # Creating the folder of the experiment.
    exp_path = ExpsPath+"Exp1_"+str(i)
    os.system("mkdir "+exp_path)
    
    JsonFileName = exp_path+"/Config_Exp1_"+str(i)+".json"
    copyfile(defaultJsonPath, JsonFileName)
    
    with open(JsonFileName, 'r+', encoding='utf-8') as f:
        data = json.load(f)

        data['simulations']['Communication_File'] = exp_path+"/Exp1_"+str(i)+"_Comm.csv"
        data['simulations']['Differential_Communication_File'] = exp_path+"/Exp1_"+str(i)+"_Diff_Comm.csv"
        data['gm_network_net1']['number_of_local_nodes'] = int(mesh[i,2])
        data['dist_algo_net1']['threshold'] = float(mesh[i,0])
        data['dist_algo_net1']['batch_size'] = int(mesh[i,1])

        f.seek(0)
        json.dump(data,f)
        f.truncate()
        
    PBSFileName = exp_path+"/Job_Exp1_"+str(i)+".pbs"
    copyfile(defaultPBSPath, PBSFileName)
        
    lines = open(PBSFileName, 'r', encoding='utf-8').readlines()
    lastline = lines[-1]
    newLastLine = lastline.split(" ")[0]+" "+"\""+JsonFileName+"\""
    lines[1] = "#PBS -N exp10000"+str(i)+"\n"
    lines[4] = "#PBS -d "+exp_path+"\n"
    del lines[-1]
    open(PBSFileName, 'w', encoding='utf-8').writelines(lines)
    open(PBSFileName, 'a', encoding='utf-8').writelines(newLastLine)

#################################################################
#################################################################

### Execute the jobs ###

# for i in range(jobs):
#     PBSFileName = newPBSPath+"Job_Exp1_"+str(i)+".pbs"
#     os.system("qsub "+str(PBSFileName))
