import torch
import torch.nn as nn
import numpy as np

def init_model(**kwargs):
    return COND_UNET(**kwargs)


"""
Modules followed by main UNet class for QPF01 CNN.

The UNet consists of an encoding pathway built from DoubleConv blocks (Conv --> BatchNorm --> ReLU)
with max pooling operators after each DoubleConv block. The bottleneck (final feature map layer) and
decoder pathway use more complex FiLM layers (Feature-wise Linear Modulation, via Perez et al. 2017:
https://arxiv.org/pdf/1709.07871) to include conditional timing information as additive and multiplicative
transforms of the encoded feature maps. At the same time, the decoder employs the UNet method of 
concatenating cached high-res versions of the encoded feature maps at each layer, preserving spatial 
fidelity. The final decoder layer performs a softmax operation to scale the output logits to 
proportion/probability space. 

NBM QPF01 processing uses model checkpoints from training on URMA 2.5km QPE01 data from 2019-2025. 
Note that some model block/modules features are optional and user-supplied. Please refer to runtime script
and training checkpoint files for appropriate parameter values and model options.
"""

class DoubleConv(nn.Module):

    # encoder path (down the U)
    # convolution, normalization, activation
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Time_Embedding(nn.Module):

    # transform time vector [day sin, day cos, hour sin, hour cos]
    # into dense embedding, imbuing non linearity
    
    def __init__(self, input_time_dim, time_embedding_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(input_time_dim, time_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(time_embedding_dim // 2, time_embedding_dim),
            nn.ReLU()
        )

    def forward(self, time_vector):
        return self.time_mlp(time_vector)

class FiLM_Layer(nn.Module):
    
    # https://arxiv.org/pdf/1709.07871
    # take as input a dense embedding representing our time vector (day/hour harmonics)
    # pass this through a linear layer to generate 2 vectors that describe our affine transformation
    # of the time info onto the feature map
    
    def __init__(self, num_features_in_layer, time_embedding_dim):
        super().__init__()
        # project time embedding into scale/shift params for this layer's channels
        self.projection = nn.Linear(time_embedding_dim, num_features_in_layer * 2)

    def forward(self, x, time_emb):
        # x: input tensor [Batch, Channels, Height, Width]
        # time_emb: dense representation of time vector [Batch, time_embedding_dim]
        
        params = self.projection(time_emb)

        # separate this tensor into 2 parameters: additive (shift) and multiplicative (scale)
        shift, scale = params.chunk(2, dim=1) 
        
        # reshape for broadcasting over H and W
        shift = shift.unsqueeze(2).unsqueeze(3)
        scale = scale.unsqueeze(2).unsqueeze(3)
        
        # apply transformation to feature map: out = (1 + scale) * x + shift
        return (1 + scale) * x + shift

class FiLM_DoubleConv(nn.Module):

    # set up Conv sequence with FiLM layers
    # FiLM layer output channels == input channels in that layer

    def __init__(self, in_channels, out_channels, time_embedding_dim, kernel_size=3, dropout_factor=0.0):
        super().__init__()
        
        self.conv_batch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.film = FiLM_Layer(out_channels, time_embedding_dim) # self.film = same # of channels as out_channels
        self.neuron_activation = nn.ReLU(inplace=True)
        self.conv_batch_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.dropout = nn.Dropout2d(p=dropout_factor)
    
    def forward(self, x, time_emb):
        # first conv
        x = self.conv_batch_1(x)
        x = self.film(x, time_emb)
        x = self.neuron_activation(x)

        #second
        x = self.conv_batch_2(x)
        x = self.film(x, time_emb)
        x = self.neuron_activation(x)

        x = self.dropout(x)
        
        return x

class COND_UNET(nn.Module):
    # conditional UNet, employing FiLM layers to encode timing information
    # avoiding huge memory/compute waste from conatenating globally static input info into initial input tensor

    # architecture is identical to normal UNet above, just with FiLM layers inserted into every layer
    
    def __init__(self, in_channels=7, out_channels=6, input_time_dim=4, time_embedding_dim=128, pos_emb_dim=16,
                 grid_size=(1597,2345),kernel_depth=[32, 64, 128, 256, 512], kernel_size=3,dropout_factor=0.0):
        super(COND_UNET, self).__init__()

        # ------ Create Time Embedding (dense vector)  ------ #
        # (4 time vars) --> tensor of length time_embedding_dim
        self.time_emb = Time_Embedding(input_time_dim, time_embedding_dim)

        # ------ Create Positional Embedding (lat/lon) ------ #
        divisor = 2**(len(kernel_depth)-1) # want grid divisible by 2 for number of pooling layers
        padded_h = int(np.ceil(grid_size[0] / divisor)) * divisor
        padded_w = int(np.ceil(grid_size[1] / divisor)) * divisor
        self.pos_emb = nn.Parameter(torch.randn(1, pos_emb_dim, padded_h, padded_w))

        total_in_channels = in_channels + pos_emb_dim

        # ------ Convolution Layers ------ #
        self.downs = nn.ModuleList()
        # the first conv layer takes the input grid (N features, H, W) and outputs feature maps with same H, W and given kernel depth
        self.downs.append(DoubleConv(total_in_channels, kernel_depth[0], kernel_size))

        #the next layers will be input/output between convolution layers
        for l_idx in range(len(kernel_depth)-2):
            self.downs.append(DoubleConv(kernel_depth[l_idx], kernel_depth[l_idx+1], kernel_size))

        # pool weights from all neurons: select max weight value from 2x2 sub grids
        # effectively reducing the spatial size of each feature map
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------ Bottleneck (bottom of the "U") ------ #
        # Same as above BUT we do not pool the weights from this layer since this is our final feature map
        # maximizing feature size and minimizing spatial size
        self.bottleneck = FiLM_DoubleConv(kernel_depth[-2], kernel_depth[-1], time_embedding_dim, kernel_size, dropout_factor)
        
        # ------ Deconvolution Layers ------ #
        self.increase_res = nn.ModuleList()
        self.deconvolve = nn.ModuleList()
        # now we reverse and decode the feature maps we just produced, working back to original grid size and # of output features
        for l_idx in range(len(kernel_depth)-1, 0, -1):
            # up sample followed by convolution, basically decoding the encoding
            self.increase_res.append(nn.ConvTranspose2d(kernel_depth[l_idx], kernel_depth[l_idx-1], kernel_size=2, stride=2))
            self.deconvolve.append(FiLM_DoubleConv(kernel_depth[l_idx], kernel_depth[l_idx-1], time_embedding_dim, kernel_size, dropout_factor))

        # ------ Final Convolution ------ #
        self.final_conv = nn.Conv2d(kernel_depth[0], out_channels, kernel_size=1)


    def forward(self, input_data, time_vector):

        batch_size = input_data.shape[0]
        pos_emb = self.pos_emb.expand(batch_size, -1, -1, -1)
        time_emb = self.time_emb(time_vector)

        x = torch.cat([input_data, pos_emb], dim=1)

        # # ------ Contraction Path ------ # #
        skip_connections = []
        for conv_block in self.downs:
            # convolution
            x = conv_block(x)
            # save feature map before reduction in spatial dimensions
            skip_connections.append(x)
            # pool
            x = self.pool(x)

        # final dense feature map, no spatial pooling
        # now we add in FiLM layers
        x = self.bottleneck(x, time_emb)
        
        # # ------ Expansion Path ------ # #
        skip_connections = skip_connections[::-1] # reverse to go back up 
        for up_step in range(len(self.increase_res)):
            # up sample
            x = self.increase_res[up_step](x)
            higher_res_map = skip_connections[up_step]

            # with padding, this shouldn't need to be used, but just in case
            if x.shape != higher_res_map.shape:
                x = nn.functional.interpolate(x, size=higher_res_map.shape[2:])

            # concat skip connection data, supplying higher res info that was lost in the pooling steps
            # which is difficult to regain upon upsampling
            concat_skip = torch.cat((higher_res_map, x), dim=1)
            # doubleConv step
            x = self.deconvolve[up_step](concat_skip, time_emb) 

        # # ------ Output ------ # #
        logits = self.final_conv(x)

        # activate
        # softmax normalizes output to 0-1, mapping back to proportion/probability space
        proportions = nn.functional.softmax(logits, dim=1)

        return proportions
