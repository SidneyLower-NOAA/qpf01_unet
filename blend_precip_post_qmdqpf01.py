#!/bin/bash
"""
Purpose: This program generates a 1-Hour Quantitative Precipitation Forecast (QPF01)
    from Quantile Mapped and Dressed (QMD) deterministic QPF06. Input to this
    program are 6-hourly forecasts of QPF06 along with precip grid constants (shape, 
    terrain, facets)

Usage: ./blend_precip_post_qmdqpf01.py

Arguments:

Input/Output Files by Env Var:
  FORT11 = QMD Precipitation Prdgen Config file (YAML)
  FORT20 = Input Model QMD forecasts file (NetCDF)
  FORT50 = Output QMD Precipitation QMD file (NetCDF)
  FORT51 = Output QMD Precipitation QMD file (GRIB2)
"""

import pandas as pd
from tqdm import tqdm
import os, sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# UNet Modules
from data_utils import process_nbm_data   #, prop_to_qpf ---- Need to refactor this to write a single file
from unet_modules import init_model


local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
s = pd.Timestamp.now()

### ------------------------- ###
###    Script Args
### -------------------------- ###
const_file = os.environ.get("FORT11")
nc_filein_list = os.environ.get("FORT20") # list of QMD QPF06 files
nc_fileout = os.environ.get("FORT50")
#grib2_fileout = os.environ.get("FORT51")

if (global_rank == 0):
    print(f"Script Args: ")
    print(f"              constants file: {const_file}")
    print(f"              NetCDF input filelist: {nc_filein_list}")
    print(f"              NetCDF output file: {nc_fileout}")

### ------------------------- ###
###  Torch/Parallel Backend
### -------------------------- ###

# Map MPI environment variables to PyTorch
#os.environ["WORLD_SIZE"] = #os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
#os.environ["RANK"] = #os.environ.get("OMPI_COMM_WORLD_RANK", "0") #rank when considering all nodes/CPUs. if running on single node, local = global
#os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0") #rank on local node

dist.init_process_group(backend="gloo")

torch.set_num_threads(1)
torch.manual_seed(42)

### ------------------------- ###
###    Model params
### -------------------------- ###
if (global_rank == 0):
    print(" ...Setting model params... ")

cnn_path = '/scratch4/STI/mdl-sti/Sidney.Lower/scripts/qmd_ml/cnn_data/'
cnn_model_filename = "IntensityW_OptDecay01_Epoch200_FiLM128_posEmb16_dropout"
saved_state = torch.load(cnn_path+cnn_model_filename+".pth", map_location='cpu')

batch_size = 1
in_channels = saved_state['model_args']['in_channels']
which_loss = saved_state['model_args']['loss']
kernel_depth = saved_state['model_args']['layers']
num_pool_layers = len(kernel_depth)-1

### ideally, these will be included in the saved state dict but
### i missed adding them in the original training
pos_emb = 16  ##saved_state['model_args']['pos_embeddings']
time_emb = 128 ##saved_state['model_args']['time_embeddings']
dropout = 0.0 ##saved_state['model_args']['dropout']

### ------------------------- ###
###      Init/load model
### -------------------------- ###
if (global_rank == 0):
    print(" ...Loading saved model... ")
UNET = init_model(in_channels=in_channels, kernel_depth=kernel_depth,pos_emb_dim=pos_emb, 
                        time_embedding_dim=time_emb, dropout_factor=dropout)
UNET.load_state_dict(saved_state['model_state_dict'])
ddp_unet = DDP(UNET)

### ------------------------- ###
###      Init data
### -------------------------- ###
if (global_rank == 0):
    print(" ...Processing data loader... ")
all_files = []
with open(nc_filein_list) as f:
    for rep in range(6):
        all_files.extend([line.rstrip() for line in f])
if (global_rank == 0):
    print(len(all_files))
nbm_loader = process_nbm_data(all_files, const_file, local_rank,
                              batch_size=batch_size,num_workers=1, 
                              num_pool_layers=num_pool_layers)

### ------------------------- ###
###        Run inference
### -------------------------- ###
if (global_rank == 0):
    print(" ...Running model... ")

print(f"       Rank {global_rank} batch = {len(nbm_loader)}")

ddp_unet.eval()
outputs_collated = []
with torch.no_grad():
    for inputs, unscaled_inputs, time_vec, valid_date in tqdm(nbm_loader):
        if (inputs is None):
            continue

        outputs = ddp_unet(inputs, time_vec)
        outputs_collated.append(outputs)

print(f" ...Rank {global_rank} pausing to collect all results... ")
dist.barrier()

f = pd.Timestamp.now()
delt = ((f - s).total_seconds()) / 60.
print(" ------------- ")
print(f"   Total runtime: {delt:.2f} minutes")
print(" ------------- ")



