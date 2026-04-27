import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import yaml
import grib2io
import tdlpackio

"""
Data processing utitlies for QPF01 CNN.

NBM_dataset class reads in NBM QPF06 NetCDF forecasts with xarray utilizing lazy loading and Pytorch's
Dataset and DataLoader classes to handle multi-processing. Input NetCDF grids are reshaped with padding
for safe handling in the UNet convolution layers. QPF values are scaled with cached statistics from the URMA
2019-2025 training dataset. Valid date information is also extracted and used as an auxillary input for the
UNet decoding layers. 

QPF01 grids are saved to NetCDF, GRIB2, and/or TDLPack.


"""


def grid_padding(image, divisor=32):

    # CONUS grid is rectangle AND a stupid, odd size (1597 x 2345)
    # so we need to implement padding to ensure theres no problems
    # in splicing together the upsampled maps with the maps from the
    # skip connections

    # divisor: number of max pooling steps, where spatial dimensions
    #          are reduced by divisor

    # input tensor will be of shape [N channels, H, W]
    h, w = image.shape[1], image.shape[2]

    new_h = int(np.ceil(h / divisor)) * divisor
    new_w = int(np.ceil(w / divisor)) * divisor

    pad_left = (new_w - w) // 2
    pad_right = (new_w - w) - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = (new_h - h) - pad_top

    # Apply padding
    padded_image = torch.nn.functional.pad(
        image, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
    )

    return padded_image


def get_grid_info(data_file):
    temp = xr.open_dataset(data_file).load()
    latitude = temp.latitude.data
    longitude = temp.longitude.data

    return latitude, longitude


def batch_collate_fn(batch):

    # Filter out the None entries
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

        qpe_stats = xr.open_dataset(
            "/scratch4/STI/mdl-sti/Sidney.Lower/data/urma/1h_qpf/training_set/stats.nc"
        ).load()
        self.precip_mean = qpe_stats.mean_log_precip.values
        self.precip_std = qpe_stats.std_log_precip.values

        consts = xr.open_dataset(const_file).load()
        terrain = consts.terrain.values
        facets = (consts.terrain_facets.sel(smooth_radius=5).values - 5.0) / 10.0
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
            features = xr.open_dataset(
                start_file,
                engine="grib2io",
                filters=dict(
                    productDefinitionTemplateNumber=8, duration=pd.Timedelta(hours=6)
                ),
            )
        except Exception:
            print(f"Cannot open file {start_file}. Skipping.")
            return None

        valid_date = pd.to_datetime(features.validDate.values)
        # why doesn't pd time delta not have an hours attr???
        lead_time = round(pd.Timedelta(features.leadTime.values).total_seconds() / 3600)
        feature_slice = features.APCP.values

        # construct mask just indicating where valid rain pixels are
        feature_mask = torch.tensor(
            np.where(feature_slice > 0.254, 1, 0), dtype=torch.int32
        ).unsqueeze(0)

        # log transform and scale QPE06
        logp1_feature = np.log1p(feature_slice)
        normalized_feature = (logp1_feature - self.precip_mean) / self.precip_std
        feature_tensor = torch.tensor(
            normalized_feature, dtype=torch.float32
        ).unsqueeze(0)

        # add timing tensors
        day_of_year = pd.Timestamp(valid_date).day_of_year
        ending_hour = int(pd.Timestamp(valid_date).hour)
        days_in_year = 366.0 if pd.Timestamp(valid_date).is_leap_year else 365.0
        sin_time = np.sin(2 * np.pi * day_of_year / days_in_year)
        cos_time = np.cos(2 * np.pi * day_of_year / days_in_year)
        ending_hour_sin = np.sin(2 * np.pi * (ending_hour / 24.0))
        ending_hour_cos = np.cos(2 * np.pi * (ending_hour / 24.0))

        # [2, H, W] (precip, mask) + [2, H, W] (terrain)
        combined_features = torch.cat(
            [feature_tensor, feature_mask, self.static_features], dim=0
        )
        # to be injected with the FiLM Layers [4, 1]
        time_vector = torch.tensor(
            [sin_time, cos_time, ending_hour_sin, ending_hour_cos], dtype=torch.float32
        )

        # add padding to ensure division by 2 all the way down
        padded_feature = grid_padding(combined_features, 2**self.n_pooling_layers)

        # lastly, we'll need to save the raw QPF06 to apply to the QPF01 proportions later
        unscaled_qpf06 = torch.tensor(feature_slice, dtype=torch.float32).unsqueeze(0)
        padded_qpf = grid_padding(unscaled_qpf06, 2**self.n_pooling_layers)

        return padded_feature, padded_qpf, time_vector, lead_time


def process_nbm_data(
    data_paths,
    constants_file_path,
    cpu_rank,
    batch_size=1,
    num_workers=1,
    num_pool_layers=5,
):

    dataset = NBM_dataset(data_paths, constants_file_path, num_pool_layers)
    data_sampler = DistributedSampler(dataset, shuffle=False, rank=cpu_rank)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=batch_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )

    return data_loader


def write_to_files(
    qpf01,
    init_date,
    lead_times,
    config,
    netcdf_fileout=None,
    grib2_fileout=None,
    tdlpack_fileout=None,
):

    qpf06_leads = lead_times.numpy()
    ny, nx = config["latitude"].shape

    # transform QPF06 leads to QPF01 leads (e.g., fill in the hours)
    qpf01_leads = []
    for lt in qpf06_leads:
        qpf01_leads.append(np.array([lt]) - np.arange(5, -1, -1))

    qpf01_leads = [pd.Timedelta(hours=item) for item in qpf01_leads]
    time_dim = pd.to_datetime(str(init_date), format="%Y%m%d%H")

    # [1 (init date), 276 (hourly lead times), ny, nx]
    qpf01_expand = qpf01.reshape(6 * len(qpf01_leads), ny, nx).unsqueeze(0)
    # and tensor --> numpy for writing
    qpf01_expand = qpf01_expand.numpy()

    # ====================================================================================
    # WRITE TO NETCDF
    # ====================================================================================
    if netcdf_fileout is not None:
        # missing val for NetCDF is -99.99
        MISSING_VAL = config["netcdf_encoding"]["missing_value"]
        qpf01_expand_netcdf = np.nan_to_num(qpf01_expand, MISSING_VAL)

        # handle meta data stuff
        lat_grid = config["latitude"]
        lon_grid = config["longitude"]
        qpf_encoding = copy.deepcopy(config["netcdf_encoding"])
        qpf_encoding["chunksizes"] = (1, 1, ny, nx)
        qpf_encoding["dtype"] = "float32"
        init_date_encoding = dict(dtype="int32")

        # write to netCDF with xarray
        ds = xr.Dataset(
            data_vars=dict(
                init_date=(
                    ["time"],
                    init_date,
                    {
                        "standard_name": "forecast_reference_time",
                        "long_name": "Model initialization date in YYYYMMDDHH format",
                    },
                ),
                qpf01=(
                    [
                        "time",
                        "lead_time",
                        "ya",
                        "xa",
                    ],
                    qpf01_expand_netcdf,
                    {
                        "standard_name": "precipitation_amount",
                        "long_name": "1-Hour Quantitative Precipitation Forecast (QPF)",
                        "units": "kg m-2",
                    },
                ),
            ),
            coords=dict(
                time=(["time"], time_dim, {"standard_name": "time"}),
                lead_time=(
                    ["lead_time"],
                    qpf01_leads,
                    {
                        "standard_name": "forecast_period",
                        "long_name": "Model forecast ending lead time",
                    },
                ),
                latitude=(
                    ["ya", "xa"],
                    lat_grid,
                    {"standard_name": "latitude", "units": "degrees_north"},
                ),
                longitude=(
                    ["ya", "xa"],
                    lon_grid,
                    {"standard_name": "longitude", "units": "degrees_east"},
                ),
            ),
        )

        ds.qpf01.encoding = qpf_encoding
        ds.init_date.encoding = init_date_encoding
        ds.to_netcdf(path=netcdf_fileout, mode="w")
        print("... Finished writing NetCDF")

    # ====================================================================================
    # WRITE TO NETCDF
    # ====================================================================================
    if grib2_fileout is not None:
        # missing val for GRIB2 is 9999.0
        MISSING_VAL = config["grib2_encoding"]["priMissingValue"]
        qpf01_expand_grib2 = np.nan_to_num(qpf01_expand, MISSING_VAL)

        g_out = grib2io.open(grib2_fileout, mode="w")
        msg = create_grib2_message(config, "qpf")
        msg.refDate = time_dim.to_pydatetime()[0]
        msg.leadTime = qpf01_leads.to_pytimedelta()[0]
        msg.data = qpf01_expand_grib2
        msg.pack()
        g_out.write(msg)

        g_out.close()
        print("... Finished writing GRIB2")

    # ====================================================================================
    # WRITE TO TDLPACK
    # ====================================================================================
    if tdlpack_fileout is not None:
        # missing val for TDLPack is 9999.0
        MISSING_VAL = config["tdlpack_encoding"]["pmiss"]
        qpf01_expand_tdlpack = np.nan_to_num(qpf01_expand, MISSING_VAL)

        t_out = tdlpackio.open(tdlpack_fileout, mode="w", format="sequential")
        ilead = [item.astype("int") for item in qpf01_leads]

        for il in range(len(ilead)):
            rec = create_tdlpack_record(
                config, "qpf", init_date, ilead[il], qpf01_expand_tdlpack[0, il]
            )
            t_out.write(rec)

        t_out.close()

        print("... Finished writing TDLPack")

    return


def create_grib2_message(cfg, msg_type):
    grib2_attrs = copy.deepcopy(cfg["grib2_encoding"])
    grib2_attrs.update(cfg[msg_type]["grib2_encoding"])
    gdtn = grib2_attrs["gridDefinitionTemplateNumber"]
    pdtn = grib2_attrs["productDefinitionTemplateNumber"]
    drtn = grib2_attrs["dataRepresentationTemplateNumber"]
    del grib2_attrs["gridDefinitionTemplateNumber"]
    del grib2_attrs["productDefinitionTemplateNumber"]
    del grib2_attrs["dataRepresentationTemplateNumber"]

    msg = grib2io.Grib2Message(gdtn=gdtn, pdtn=pdtn, drtn=drtn)
    msg.section3[5:] = grib2_attrs["gridDefinitionTemplate"]
    del grib2_attrs["gridDefinitionTemplate"]
    for k, v in grib2_attrs.items():
        setattr(msg, k, v)
    msg.bitMapFlag = grib2io.templates.Grib2Metadata(255, table="6.0")

    return msg


def read_yaml_config(paths, domain):
    tmp = {}
    for path in paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            if data:
                tmp.update(data)
    grib2_domain_grid = copy.deepcopy(tmp["grids"][domain])
    del tmp["grids"]
    config = copy.deepcopy(tmp)
    config["grib2_encoding"].update(grib2_domain_grid)
    return config


def create_tdlpack_record(
    cfg, msg_type, idate, ilead, data, thresh=0.0, pct=0, scale=0.0
):
    tdlp_attrs = copy.deepcopy(cfg["tdlpack_encoding"])
    tdlp_attrs.update(cfg[msg_type]["tdlpack_encoding"])

    id1, id2, id4 = 0, 0, 0
    id3 = ilead

    IN_TO_MM = 25.4
    MM_TO_IN = 1.0 / IN_TO_MM

    griddef_co = {
        "mapProjection": 3,
        "nx": 2345,
        "ny": 1597,
        "latitudeLowerLeft": 19.2290,
        "longitudeLowerLeft": 126.2766,
        "standardLatitude": 25.0000,
        "orientationLongitude": 95.0000,
        "gridLength": 2539.702881,
    }

    if msg_type != "qpf":
        print("[DATA_UTILS/TDLPACK]: CAN ONLY WRITE OUT QPF RECORDS.")
    else:
        id1 = (tdlp_attrs["cccfff"] * 1000) + tdlp_attrs["dd"]
        plain = tdlp_attrs["plain"].format(pct=pct)
        data = np.where(data >= 0, data * MM_TO_IN, data)

    recid = [id1, id2, id3, id4]

    rec = tdlpackio.TdlpackRecord(type="grid")

    # grid def, section 3
    for item, val in griddef_co.items():
        setattr(rec, item, val)

    rec.name = plain
    rec.refDate = pd.to_datetime(idate, format=("%Y%m%d%H"))
    rec.id = recid
    rec.decScaleFactor = tdlp_attrs["decimal_scale_factor"]
    rec.data = data

    rec.pack()

    return rec
