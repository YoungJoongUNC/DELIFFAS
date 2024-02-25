
#############################################################################################
# Imports
#############################################################################################

from Utils.positional_encoding import *
from Utils.main_model.nerf_model import *
import os
import numpy as np

from Libs.Networks import make_encoder
from Libs.Networks import make_discriminator
import pdb
#############################################################################################
# Create Nerf
#############################################################################################

def create_nerf(args, textureFeatureSize, positionalPose, numberOfSmallNetworks):


    uv_embed_fn, input_ch = get_embedder(args.multires, args.i_embed, inputDims=2)

    skips =  []
    numSkips = args.netdepth // 4
    for i in range(1, numSkips):
        skips.append(i*4)
    print('Nerf skip: ', skips, flush=True)

    ]
    print('initializing Light Field Network')

    grad_vars = []
    if args.use_adversarial_loss:
        disc_vars = []
    models = {}
    modelList = []

    for n in range (0, numberOfSmallNetworks):
        modelN = init_nerf_model(
                                D                       = args.netdepth,
                                W                       = args.netwidth,
                                uv_feat_ch              = 2*input_ch,
                                normal_feat_ch          = 2*args.textureFeatureSize,
                                output_ch               = 3, #output_ch,
                                skips                   = []) #skips

        grad_vars += modelN.trainable_variables
        models['model_' + str(n)] = modelN
        modelList.append(modelN)


    ########################
    # Generator
    ########################
    print('initializing feature encoders')

    # outer jelly
    featureExtractor_out = make_encoder(args, 9, textureFeatureSize, 1024, 1024, args.net_mode)
    grad_vars += featureExtractor_out.model.trainable_weights
    models['featureExtractor_out'] = featureExtractor_out.model

    # original jelly
    featureExtractor_orig = make_encoder(args, 9, textureFeatureSize, 1024, 1024, args.net_mode)
    grad_vars += featureExtractor_orig.model.trainable_weights
    models['featureExtractor_orig'] = featureExtractor_orig.model



    ########################
    # Discriminator
    ########################
    if args.use_adversarial_loss:
        if args.fg_loss_only:
            new_H = 512
            new_W = 512
        else:
            new_H = args.resolutionV
            new_W = args.resolutionU
        discriminator = make_discriminator(args, new_W, new_H, 3)  # argument order: U-V

        disc_vars += discriminator.model.trainable_weights
        models['discriminator'] = discriminator.model

    print('initializing discriminator')

    ########################
    # query function
    ########################
    def network_query_fn(out_uv, in_uv, outNormalFeature, origNormalFeature, network_fn):


        out_uv_embedded = uv_embed_fn(out_uv)
        in_uv_embedded = uv_embed_fn(in_uv)

        Features = tf.concat([out_uv_embedded, in_uv_embedded, outNormalFeature, origNormalFeature],-1)


        return network_fn(Features)


    ########################
    # recover saved checkpoint

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.load_best_model:
        start = -1
        with open(os.path.join(basedir, expname, 'best_psnr.txt'), 'r') as f:
            last_line = f.readlines()[-1]
            start = int(last_line.split('_')[1].split(' :')[0]) + 1

        print('Found ckpts', flush=True)

        # set the iteration step
        ft_weights = os.path.join(basedir, expname, 'model_0_best_psnr.npy')

        print('Resetting step to', start, flush=True)

        # load the coarse net weights

        for n in range(0, numberOfSmallNetworks):
            ft_weights = os.path.join(basedir, expname, 'model_0_best_psnr.npy')
            print('Reloading from', ft_weights, flush = True)
            models['model_' + str(n)].set_weights(np.load(ft_weights, allow_pickle=True))


        # load the feature extractor net weights (outer jelly)
        ft_weights_feature_out = os.path.join(basedir, expname, 'featureExtractor_out_best_psnr.npy')
        print('Reloading from', ft_weights_feature_out, flush=True)
        featureExtractor_out.model.set_weights(np.load(ft_weights_feature_out, allow_pickle=True))


        # load the feature extractor net weights (original jelly)
        ft_weights_feature_orig = os.path.join(basedir, expname, 'featureExtractor_orig_best_psnr.npy')
        print('Reloading from', ft_weights_feature_orig, flush=True)
        featureExtractor_orig.model.set_weights(np.load(ft_weights_feature_orig, allow_pickle=True))


        if args.use_adversarial_loss:
            ft_discriminator = os.path.join(basedir, expname, 'discriminator_best_psnr.npy')
            print('Reloading from', ft_discriminator, flush=True)
            discriminator.model.set_weights(np.load(ft_discriminator, allow_pickle=True))
    else:
        if args.ft_iteration is not None and args.ft_iteration != 'None':
            specifidIteration = str(args.ft_iteration).zfill(7)
        else:
            specifidIteration = '_'

        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('model_' in f and 'best' not in f and 'fine' not in f and 'gcnn' not in f and 'optimizer' not in f and specifidIteration in f)]

        if len(ckpts) > 0 and not args.no_reload:

            print('Found ckpts', flush=True)

            # set the iteration step
            ft_weights = ckpts[-1]
            start = int(ft_weights[-11:-4]) + 1
            print('Resetting step to', start, flush=True)

            # load the coarse net weights

            for n in range(0, numberOfSmallNetworks):
                ft_weights = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('model_'+ str(n) in f and 'best' not in f and 'fine' not in f and 'gcnn' not in f and 'optimizer' not in f and specifidIteration in f)][-1]
                print('Reloading from', ft_weights, flush = True)

                models['model_' + str(n)].set_weights(np.load(ft_weights, allow_pickle=True))



            # load the feature extractor net weights (outer jelly)
            ckptsFeature_out = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('featureExtractor_out' in f and 'best' not in f and specifidIteration in f)]
            ft_weights_feature_out = ckptsFeature_out[-1]
            print('Reloading from', ft_weights_feature_out, flush=True)
            featureExtractor_out.model.set_weights(np.load(ft_weights_feature_out, allow_pickle=True))


            # load the feature extractor net weights (original jelly)
            ckptsFeature_orig = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('featureExtractor_orig' in f and 'best' not in f and specifidIteration in f)]
            ft_weights_feature_orig = ckptsFeature_orig[-1]
            print('Reloading from', ft_weights_feature_orig, flush=True)
            featureExtractor_orig.model.set_weights(np.load(ft_weights_feature_orig, allow_pickle=True))



    if not args.use_adversarial_loss:
        disc_vars = []
        discriminator=None
    embedMotion_fn=None
    return network_query_fn, modelList, start, grad_vars, disc_vars, models, embedMotion_fn, discriminator

