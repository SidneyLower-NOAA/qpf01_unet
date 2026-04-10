# qpf01_unet


Conditional UNet to infer hourly QPF proportions from NBM v5.0 QPF06.
Built with Pytorch, includes Conv2D layers, conditional embeddings, and skip connections.

## Scripts

- data_utils: custom torch dataset classes compatible with xarray IO and lazy loading with torch's DataLoader
- unet_modules: convolution layers/model components
- blend_precip_post_qmdqpf01: runtime script to run inference using pre-trained model state

## Requirements / Dependencies

- python 3.13
- pytorch CPU 2.9
- xarray 2025.12
- grib2io 2.7
- numpy, pandas