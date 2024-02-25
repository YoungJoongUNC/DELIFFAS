import os
import imp
import pdb

def make_discriminator(args, inputResolutionU, inputResolutionV, in_ch):
    module = args.discriminator_module
    path = args.discriminator_path

    discriminator = imp.load_source(module, path).Discriminator(inputResolutionV, inputResolutionU, in_ch)

    return discriminator