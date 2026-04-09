#!/bin/bash
"""
Purpose: This program generates a 1-Hour Quantitative Precipitation Forecast (QPF01)
    from Quantile Mapped and Dressed (QMD) deterministic QPF06. Input to this
    program are 6-hourly forecasts of QPF06 along with precip grid constants (shape, 
    terrain, facets)

Usage: ./blend_precip_post_qmdqpf01.py

Arguments:

Input/Output Files by Env Var:
  FORT11 = QMD Precipitation constants file (NetCDF)
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
from data_utils import get_grid_info, process_nbm_data, write_to_NetCDF, write_to_GRIB2
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
grib2_fileout = os.environ.get("FORT51")


init_date = int(sys.argv[1])

if (global_rank == 0):
    print(f"Script Args: ")
    print(f"              constants file: {const_file}")
    print(f"              NetCDF input filelist: {nc_filein_list}")
    print(f"              NetCDF output file: {nc_fileout}")

### ------------------------- ###
###  Torch/Parallel Backend
### -------------------------- ###

dist.init_process_group(backend="gloo")
torch.set_num_threads(4)
torch.manual_seed(42)

### ------------------------- ###
###    Model params
### -------------------------- ###
if (global_rank == 0):
    print(" ...Setting model params... ")

cnn_path = '/scratch4/STI/mdl-sti/Sidney.Lower/scripts/qmd_ml/cnn_data/'
cnn_model_filename = "ConditionalUNet_KLDiv_v1_CONT2_AMSEFineTune"
saved_state = torch.load(cnn_path+cnn_model_filename+".pth", map_location='cpu')

batch_size = 1
num_workers=1
in_channels = saved_state['model_args']['in_channels']
which_loss = saved_state['model_args']['loss']
kernel_depth = saved_state['model_args']['layers']
num_pool_layers = len(kernel_depth)-1
pos_emb = saved_state['model_args']['pos_embeddings']
time_emb = saved_state['model_args']['time_embeddings']
dropout = 0.0


#grid info
latitude, longitude = get_grid_info(const_file)
ny, nx = latitude.shape


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
    all_files.extend([line.rstrip() for line in f])
    
if (global_rank == 0):
    print(f"Total files: {len(all_files)}")

nbm_loader = process_nbm_data(all_files, const_file, local_rank,
                              batch_size=batch_size,num_workers=num_workers, 
                              num_pool_layers=num_pool_layers)

n_per_rank = len(nbm_loader)
print(f"       Rank {global_rank} handling {n_per_rank} lead times.")
### ------------------------- ###
###        Run inference
### -------------------------- ###
if (global_rank == 0):
    print(" ...Running model... ")

ddp_unet.eval()
output_shape = (6, ny, nx)
outputs_collated = torch.zeros((n_per_rank, *output_shape))
lead_times_collated = torch.zeros((n_per_rank))
b=0
with torch.no_grad():
    for inputs, nbm_qpf06, time_vector, nbm_qpf06_lead_time in tqdm(nbm_loader):
        if (inputs is None):
            continue

        # generate QPF01 proportions from trained CNN
        outputs_qpf01_prop = ddp_unet(inputs, time_vector)
        # get QPF01 amounts and remove padding from tensors
        outputs_qpf01 = proportions_to_qpf(outputs_qpf01_prop.detach(), qpf06.detach(), ny, nx)

        #collect local outputs
        outputs_collated[b] = outputs_qpf01
        lead_times_collated[b] = nbm_qpf06_lead_time
        b+=1


### ------------------------- ###
###      Save outputs
### -------------------------- ###
if (global_rank == 0):
    print("... Gathering outputs from all ranks ...")
output_gather_list = [torch.zeros_like(outputs_collated) for _ in range(dist.get_world_size())]
lead_time_gather_list = [torch.zeros_like(lead_times_collated) for _ in range(dist.get_world_size())]


# Collect everything
dist.barrier()
dist.all_gather(output_gather_list, outputs_collated)
dist.all_gather(lead_time_gather_list, lead_times_collated)

# 4. Rank 0 saves the file
if (global_rank == 0):
    print("... Saving QPF01 to NetCDF & GRIB2...")

    ## [16, 3, 6, Y, X] -> [48, 6, Y, X], but we only need the 46 valid lead times
    all_outputs = torch.cat(output_gather_list, dim=0)[:46]
    ## [16, 3] --> [48]
    all_leads = torch.cat(lead_time_gather_list, dim=0)[:46]
    
    write_to_NetCDF(all_outputs, latitude, longitude, init_date, all_leads, nc_fileout)
    write_to_GRIB2(model_output, qmd_qpf06, lat_grid, lon_grid, init_date, lead_times, output_path)


f = pd.Timestamp.now()
delt = ((f - s).total_seconds()) / 60.
print(" ------------- ")
print(f"   Total runtime: {delt:.2f} minutes")
print(" ------------- ")



