
import run_nerf
from Utils.arg_parser import config_parser
import sys
from Libs.Runner import make_runner
import pdb
import imp


#############################################################################################
# Main
#############################################################################################

if __name__ == '__main__':

    args = config_parser().parse_args()
    print('START', flush = True)

    nerfRunner = make_runner(args)

    if nerfRunner.args.render_only:
        nerfRunner.test()
    else:
        nerfRunner.train()

    print('END', flush = True)