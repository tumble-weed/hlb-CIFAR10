import torch
#############################################
#            Network Definition             #
#############################################

scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
    'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 10
}

class SpeedyResNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['project'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['residual1'](x)
        x = self.net_dict['residual2'](x)
        x = self.net_dict['residual3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        x = self.net_dict['temperature'](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(x.shape[0]//2, dim=0)
            x = .5 * orig + .5 * flipped
        return x


def make_net():
    # TODO: A way to make this cleaner??
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    whiten_conv_depth = 3*hyp['net']['whitening']['kernel_size']**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': Conv(3, whiten_conv_depth, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0),
            'project': Conv(whiten_conv_depth, depths['init'], kernel_size=1, norm=2.2), # The norm argument means we renormalize the weights to be length 1 for this as the power for the norm, each step
            'activation': nn.GELU(),
        }),
        'residual1': ConvGroup(depths['init'],   depths['block1'], hyp['net']['conv_norm_pow']),
        'residual2': ConvGroup(depths['block1'], depths['block2'], hyp['net']['conv_norm_pow']),
        'residual3': ConvGroup(depths['block2'], depths['block3'], hyp['net']['conv_norm_pow']),
        'pooling': FastGlobalMaxPooling(),
        'linear': Linear(depths['block3'], depths['num_classes'], bias=False, norm=5.),
        'temperature': TemperatureScaler(hyp['opt']['scaling_factor'])
    })

    net = SpeedyResNet(network_dict)
    net = net.to(hyp['misc']['device'])
    net = net.to(memory_format=torch.channels_last) # to appropriately use tensor cores/avoid thrash while training
    net.train()
    net.half() # Convert network to half before initializing the initial whitening layer.


    ## Initialize the whitening convolution
    with torch.no_grad():
        # Initialize the first layer to be fixed weights that whiten the expected input values of the network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
        init_whitening_conv(net.net_dict['initial_block']['whiten'],
                            data['train']['images'].index_select(0, torch.randperm(data['train']['images'].shape[0], device=data['train']['images'].device)),
                            num_examples=hyp['net']['whitening']['num_examples'],
                            pad_amount=hyp['net']['pad_amount'],
                            whiten_splits=5000) ## Hardcoded for now while we figure out the optimal whitening number
                                                ## If you're running out of memory (OOM) feel free to decrease this, but
                                                ## the index lookup in the dataloader may give you some trouble depending
                                                ## upon exactly how memory-limited you are

        ## We initialize the projections layer to return exactly the spatial inputs, this way we start
        ## at a nice clean place (the whitened image in feature space, directly) and can iterate directly from there.
        torch.nn.init.dirac_(net.net_dict['initial_block']['project'].weight)

        for layer_name in net.net_dict.keys():
            if 'residual' in layer_name:
                ## We do the same for the second layer in each residual block, since this only
                ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
                ## convolution. This can be easily scaled down by the network, and the weights can more easily
                ## pivot in whichever direction they need to go now.
                torch.nn.init.dirac_(net.net_dict[layer_name].conv2.weight)

    return net
