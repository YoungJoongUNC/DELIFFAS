import os
import imp
import pdb

def make_encoder(args, in_ch, out_ch, inputResolutionU, inputResolutionV, netMode):
    module = args.encoder_module
    path = args.encoder_path

    encoder = imp.load_source(module, path).UNet(in_ch, out_ch, inputResolutionU, inputResolutionV, netMode=netMode, deep_skip_unet=args.deep_skip_unet)

    return encoder