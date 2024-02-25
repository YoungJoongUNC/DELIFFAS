import os
import imp
import pdb
import sys
def make_runner(args):
    module = args.runner_module
    path = args.runner_path

    nerfRunner = imp.load_source(module, path).Nerf(args)

    return nerfRunner