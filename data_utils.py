import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

"""
Data processing utitlies for QPF01 CNN.

NBM_dataset class reads in NBM QPF06 NetCDF forecasts with xarray utilizing lazy loading and Pytorch's
Dataset and DataLoader classes to handle multi-processing. Input NetCDF grids are reshaped with padding
for safe handling in the UNet convolution layers. QPF values are scaled with cached statistics from the URMA
2019-2025 training dataset. Valid date information is also extracted and used as an auxillary input for the
UNet decoding layers. 

(((to do))): QPF01 grids are saved to NetCDF and GRIB2.


"""


def grid_padding(image, divisor=32):

    # CONUS grid is rectangle AND a stupid, odd size (1597 x 2345)
    # so we need to implement padding to ensure theres no problems
    # in splicing together the upsampled maps with the maps from the 
    # skip connections

    # divisor: number of max pooling steps, where spatial dimensions
    #          are reduced by divisor

    #input tensor will be of shape [N channels, H, W]
    h, w = image.shape[1], image.shape[2]
    
    new_h = int(np.ceil(h / divisor)) * divisor
    new_w = int(np.ceil(w / divisor)) * divisor

    pad_left = (new_w - w) // 2
    pad_right = (new_w - w) - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = (new_h - h) - pad_top
    
    # Apply padding
    padded_image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    
    return padded_image


def get_grid_info(data_file):
    temp = xr.open_dataset(data_file).load()
    latitude = temp.latitude.data
    longitude = temp.longitude.data

    return latitude, longitude



def batch_collate_fn(batch):
    
    #Filter out the None entries
    batch = [item for item in batch if item is not None]
    
    # Check if the batch is now empty (all samples were bad)
    if not batch:
        return None, None, None, None 
    
    return torch.utils.data.dataloader.default_collate(batch)


class NBM_dataset(Dataset):
    def __init__(self, qpf06_files, const_file, num_pool):

        # 6h precip
        self.fpath = qpf06_files
        self.n_samples = len(qpf06_files)
        self.n_pooling_layers = num_pool

        qpe_stats = xr.open_dataset("/scratch4/STI/mdl-sti/Sidney.Lower/data/urma/1h_qpf/training_set/stats.nc").load()
        self.precip_mean = qpe_stats.mean_log_precip.values
        self.precip_std = qpe_stats.std_log_precip.values

        consts = xr.open_dataset(const_file).load()
        terrain = consts.terrain.values
        facets = (consts.terrain_facets.sel(smooth_radius=5).values - 5.)/10.
        normalized_terrain = (terrain - np.nanmean(terrain)) / np.nanmax(terrain)

        terrain_tensor = torch.from_numpy(normalized_terrain).float().unsqueeze(0)
        facets_tensor = torch.from_numpy(facets).float().unsqueeze(0)

        # shape = [2, height, width]
        self.static_features = torch.cat([terrain_tensor, facets_tensor], dim=0)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        start_file = self.fpath[idx]

        try:
            #features = xr.open_dataset(start_file, engine='grib2io', filters=dict(productDefinitionTemplateNumber=8))
            features = xr.open_dataset(start_file,engine='grib2io', filters=dict(productDefinitionTemplateNumber=8, duration=pd.Timedelta(hours=6)))
        except:
            print("file corrupted / doesn't exist. skipping")
            return None
            
        valid_date = pd.to_datetime(features.validDate.values)
        # why doesn't pd time delta not have an hours attr???
        lead_time = round(pd.Timedelta(features.leadTime.values).total_seconds() / 3600)
        feature_slice = features.APCP.values
    
        # construct mask just indicating where valid rain pixels are
        feature_mask = torch.tensor(np.where(feature_slice > 0.254, 1, 0), dtype=torch.int32).unsqueeze(0)

        # log transform and scale QPE06
        logp1_feature = np.log1p(feature_slice)
        normalized_feature = (logp1_feature - self.precip_mean) / self.precip_std
        feature_tensor = torch.tensor(normalized_feature, dtype=torch.float32).unsqueeze(0)

        # add timing tensors
        day_of_year = pd.Timestamp(valid_date).day_of_year
        ending_hour = int(pd.Timestamp(valid_date).hour)
        days_in_year = 366.0 if pd.Timestamp(valid_date).is_leap_year else 365.0
        sin_time = np.sin(2 * np.pi * day_of_year / days_in_year)
        cos_time = np.cos(2 * np.pi * day_of_year / days_in_year)
        ending_hour_sin = np.sin(2*np.pi * (ending_hour / 24.))
        ending_hour_cos = np.cos(2*np.pi * (ending_hour / 24.))
        
        # [2, H, W] (precip, mask) + [2, H, W] (terrain)
        combined_features = torch.cat([feature_tensor, feature_mask, self.static_features], dim=0)
        # to be injected with the FiLM Layers [4, 1]
        time_vector = torch.tensor([sin_time, cos_time, ending_hour_sin, ending_hour_cos], dtype=torch.float32)

        # add padding to ensure division by 2 all the way down
        padded_feature = grid_padding(combined_features, 2**self.n_pooling_layers)

        # lastly, we'll need to save the raw QPF06 to apply to the QPF01 proportions later
        unscaled_qpf06 = torch.tensor(feature_slice, dtype=torch.float32).unsqueeze(0)
        padded_qpf = grid_padding(unscaled_qpf06, 2**self.n_pooling_layers)


        return padded_feature, padded_qpf, time_vector, lead_time


def process_nbm_data(data_paths, constants_file_path, cpu_rank, batch_size=1, num_workers=1, num_pool_layers=5):
    
    dataset = NBM_dataset(data_paths, constants_file_path, num_pool_layers)
    data_sampler = DistributedSampler(dataset, shuffle=False, rank=cpu_rank)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,sampler=data_sampler,
                              collate_fn=batch_collate_fn,num_workers=num_workers,pin_memory=False)


    return data_loader


def proportions_to_qpf(model_output, qmd_qpf06, ny, nx):


    # model output shape is [N lead times, 6, H_padded, W_padded]
    # remove padded pixels
    model_output_to_conus = model_output[:,:,:ny,:nx]
    qpf06_to_conus = qmd_qpf06[:,:,:ny,:nx]

    # proportions --> QPF01
    qpf01 = model_output_to_conus * qpf06_to_conus

    # shape [N lead times, 6 hourly amounts, ny, nx]
    return qpf01



"""

TO DO:::::::

make sure we're conforming to the prodgen/grid yaml config files...
look at Eric's qpf prodgen script to see how he wrote the grib2
can we not use xarray to write?







"""
    

def write_to_NetCDF(qpf01, lat_grid, lon_grid, init_date, lead_times, output_path):

    qpf06_leads = lead_times.numpy()

    # transform QPF06 leads to QPF01 leads (e.g., fill in the hours)
    qpf01_leads = []
    for lt in qpf06_leads:
        qpf01_leads.append(np.array([lt]) - np.arange(5, -1, -1))

    qpf01_leads = [pd.Timedelta(hours=item) for item in qpf01_leads]

    # [1 (init date), 276 (hourly lead times), ny, nx]
    qpf01_expand = qpf01.reshape(6*len(qp01_leads), ny, nx).unsqueeze(0)

    time_dim = pd.to_datetime(str(init_date), format="%Y%m%d%H")

    #write to zarr via xarray dataset
    ds = xr.Dataset(data_vars=dict(
                                init_date=(["time"], 
                                           init_date, 
                                           {"standard_name": "forecast_reference_time", 
                                            "long_name": "Model initialization date in YYYYMMDDHH format"}),
                                pred_qpe01=(["time","lead_time", "ya", "xa",], 
                                            qpf01_expand, 
                                            {"standard_name": "precipitation_amount", 
                                             "long_name": "1-Hour Quantitative Precipitation Forecast (QPF)", "units": "kg m-2"})),
                    
                    coords=dict(
                                time=(["time"], time_dim, {"standard_name": "time"}),
                                lead_time=(["lead_time"], qpf01_leads, {"standard_name": "forecast_period", "long_name": "Model forecast ending lead time"}),
                                latitude=(["ya", "xa"], lat_grid, {"standard_name": "latitude", "units": "degrees_north"}),
                                longitude=(["ya", "xa"], lon_grid, {"standard_name": "longitude", "units": "degrees_east"})),

    
    ds.to_netcdf(output_path,mode='w')
    
    return

def write_to_GRIB2(model_output, qmd_qpf06, lat_grid, lon_grid, valid_dates, output_path):


    # model output shape is [N lead times, 6, H_padded, W_padded]
    nleads = np.shape(model_output)[0]
    ny, nx = np.shape(lat_grid)
    
    qpf06[qpf06 <= 0.254] = 0.0

    # remove padded pixels
    model_output_to_conus = model_output[:,:,:ny,:nx]
    qpf06_to_conus = qmd_qpf06[:,:,:ny,:nx]

    # proportions --> QPF01
    qpf01 = model_output_to_conus.cpu().numpy() * torch.expm1(qpf06_to_conus).cpu().numpy()
    qpf01 = np.nan_to_num(qpf01, 0.0)
    qpf01_expand = qpf01.reshape(6*nleads, ny, nx)
    
    valid_dates_to_datetime = []
    for b in valid_dates:
        for i in range(5, -1, -1):
            valid_dates_to_datetime.append(pd.to_datetime(b, format="%Y%m%d%H") - pd.Timedelta(hours=i))


    #write to zarr via xarray dataset
    ds = xr.Dataset(data_vars=dict(pred_qpe01=(["validDate","y", "x",], new_pred01)),
                       coords=dict(longitude=(["y", "x"], lon),
                                   latitude=(["y", "x"], lat),
                                   validDate=valid_dates_datetime,
                                  leadTime=pd.Timedelta(hours=lead_time)),
                      )

    ds.validDate.encoding['units'] = 'nanoseconds since 1970-01-01'
    ds.validDate.encoding['dtype'] = "int64"
    ds.to_zarr(output_path+f'/qpf01_nbmf{lead_time:03d}_R{gpu_rank}_batch{batch_number}.zarr',
                 mode='w', consolidated=True,zarr_format=2)
    
    return
