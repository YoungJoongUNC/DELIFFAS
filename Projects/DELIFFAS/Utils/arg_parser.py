
import configargparse

#############################################################################################
# Config parser
#############################################################################################

def config_parser():

    parser = configargparse.ArgumentParser()

    # config and experiment name
    parser.add_argument('--config',             is_config_file=True,                                                            help='Config file path')
    parser.add_argument("--expname",            type=str,                                                                       help='Experiment name')

    # file paths
    parser.add_argument("--basedir",            type=str,           default='./logs/',                                          help='Where to store ckpts and logs')
    parser.add_argument("--datadir",            type=str,           default='',                                                 help='Input data directory')
    parser.add_argument("--imagedir",           type=str,           default='',                                                 help='Input image directory')
    parser.add_argument("--fgmaskdir",          type=str,           default='',                                                 help='Input foreground mask directory')
    parser.add_argument("--meshesdir",          type=str,           default='',                                                 help='Input meshes directory')
    parser.add_argument("--mesh_seq_path",          type=str,           default='',                                                 help='Input meshes directory')
    parser.add_argument("--texdir",             type=str,           default='',                                                 help='Input tex directory')
    parser.add_argument("--depthdir",           type=str,           default='',                                                 help='Input depth directory')
    parser.add_argument("--meshfilepath",       type=str,           default='',                                                 help='Input mesh file')
    parser.add_argument("--charFilePath",       type=str,           default='',                                                 help='Character mesh file')
    parser.add_argument("--testsavedir",        type=str,           default='',                                                 help='Optional path for saving results')
    parser.add_argument("--calibdir",           type=str,           default='',                                                help='Input camera calibration directory')
    #mesh_seq_path
    # training
    parser.add_argument("--multiProcessing",                                            action='store_true',                    help='Use multiprocessing for faster data loading')
    parser.add_argument("--multi_cores",        type=int,           default=16,                                                 help='Number of cores running in parallel')
    parser.add_argument("--N_rand",             type=int,           default=1024,                                               help='Batch size (number of random rays per gradient step)')
    parser.add_argument("--num_views_per_batch",type=int,           default=8,                                                  help='Number of views to sample per batch')
    parser.add_argument("--queue_size",         type=int,           default=10,                                                 help='Buffer size for the loading queue')
    parser.add_argument("--lrate",              type=float,         default=5e-4,                                               help='Learning rate')
    parser.add_argument("--lrate_decay",        type=int,           default=800,                                                help='Exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk",              type=int,           default=1024,                                               help='Number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--random_seed",        type=int,           default=None,                                               help='Fix random seed for repeatability')
    parser.add_argument("--perturb",            type=float,         default=1.,                                                 help='Set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--raw_noise_std",      type=float,         default=0.,                                                 help='Std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--dilate_mask",                                                action='store_true',                    help='Define whether using the image mask to constrain the sampling')
    parser.add_argument("--train_cameras",      type=int,                               action='append',                        help='Training cameras')
    parser.add_argument("--test_cameras",       type=int,                               action='append',                        help='Testing cameras')
    parser.add_argument("--validation_cameras", type=int,                               action='append',                        help='Validation cameras')

    # slurm
    parser.add_argument("--timeLimit",          type=str,           default='1-00:00:00',                                       help='Maximum time for slurm job',                                                              required=True)
    parser.add_argument("--slurmArrayID",       type=int,           default=0,                                                  help='Job array id')
    parser.add_argument("--slurmID",            type=int,           default=0,                                                  help='Slurm job id')

    # architeture
    parser.add_argument("--multiNet",                                                   action='store_true',                    help='Use a MLP per body part')
    parser.add_argument("--netdepth",           type=int,           default=16,                                                 help='Layers in network')
    parser.add_argument("--netwidth",           type=int,           default=128,                                                help='Channels per layer')
    parser.add_argument("--depth",                                                      action='store_true',                    help='Use depth estimated from geometry')
    parser.add_argument("--N_samples",          type=int,           default=32,                                                 help='Number of coarse samples per ray')
    parser.add_argument("--i_embed",            type=int,           default=0,                                                  help='Set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires",           type=int,           default=10,                                                 help='Log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views",     type=int,           default=4,                                                  help='Log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--use_viewdirs",                                               action='store_true',                    help='Use full 5D input instead of 3D')
    parser.add_argument("--positional_pose",                                            action='store_true',                    help='Apply positional encoding on the pose vector')
    parser.add_argument("--textureFeatureSize", type=int,           default=32,                                                 help='Texture feature size')
    parser.add_argument("--viewDirFeatureSize", type=int,           default=32,                                                 help='Texture feature size')

    # misc
    parser.add_argument("--calib",              type=str,                                                                       help='Calibration file name',                                                                   required=True)
    parser.add_argument("--ft_iteration",       type=int,           default=None,                                               help='Specific iteration snapshot')
    parser.add_argument("--no_reload",                                                  action='store_true',                    help='Do not reload weights from saved ckpt')
    parser.add_argument("--resolutionU",        type=int,           default=1285,                                               help='Training U resolution')
    parser.add_argument("--resolutionV",        type=int,           default=940,                                                help='Training V resolution')
    parser.add_argument("--train_start_frame",  type=int,           default=None,                                               help='Start frame for training',                                                                required=True)
    parser.add_argument("--train_end_frame",    type=int,           default=None,                                               help='End frame for training',                                                                  required=True)
    parser.add_argument("--render_only",                                                action='store_true',                    help='Do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--iteration",          type=int,           default=0,                                                  help='Number of recursive refinement steps')
    parser.add_argument("--write_pc",                                                   action='store_true',                    help='Write the pc file at test time')
    parser.add_argument("--use_gt_tex",                                                 action='store_true',                    help='Use the GT Tex generated by Alldieck et al.')
    parser.add_argument("--create_pc",                                                  action='store_true',                    help='Create pointcloud')
    parser.add_argument("--moving_cam",                                                 action='store_true',                    help='Moving camera file')
    parser.add_argument("--quantitative_eval",                                          action='store_true',                    help='Quantitative evaluation mode')

    # dataset / logging /saving options
    parser.add_argument("--i_print",            type=int,           default=100,                                                help='Frequency of console printout and metric logging')
    parser.add_argument("--i_img",              type=int,           default=5000,                                               help='Frequency of tensorboard image logging')
    parser.add_argument("--i_weights",          type=int,           default=10000,                                              help='Frequency of weight ckpt saving')

    # temporal batch
    parser.add_argument("--N_batch",          type=int,           default=4,                                                 help='Number of temporal frames for batch processing')

    # module path
    parser.add_argument("--runner_module", type=str, default='', help='Where is run_nerf.py')
    parser.add_argument("--runner_path", type=str, default='', help='Where is run_nerf.py')
    parser.add_argument("--encoder_module", type=str, default='', help='Where is feature extractor')
    parser.add_argument("--encoder_path", type=str, default='', help='Where is feature extractor')
    parser.add_argument("--discriminator_module", type=str, default='', help='Where is discriminator')
    parser.add_argument("--discriminator_path", type=str, default='', help='Where is discriminator')


    # loss
    parser.add_argument("--use_adversarial_loss",                                        action='store_true',                    help='Use the adversarial loss.')
    parser.add_argument("--w_adversarial_loss",            type=float,         default=0.001,                                     help='Set to adjust adversarial loss weight')
    parser.add_argument("--use_perceptual_loss",                                         action='store_true',                    help='Use the perceptual loss.')
    parser.add_argument("--w_perceptual_loss",            type=float,         default=0.01,                                       help='Set to adjust perceptual loss weight')

    parser.add_argument("--use_ssim_loss",                                               action='store_true',                    help='Use the structural similarity loss.')
    parser.add_argument("--w_ssim_loss",            type=float,         default=1.0,                                       help='Set to adjust structural similarity loss weight')


    parser.add_argument("--use_tex_adversarial_loss",                                        action='store_true',                    help='Use the adversarial loss.')
    parser.add_argument("--w_tex_adversarial_loss",            type=float,         default=0.001,                                     help='Set to adjust adversarial loss weight')
    parser.add_argument("--use_tex_ssim_loss",                                               action='store_true',                    help='Use the structural similarity loss.')
    parser.add_argument("--w_tex_ssim_loss",            type=float,         default=1.0,                                       help='Set to adjust structural similarity loss weight')

    parser.add_argument("--net_mode", type=str, default='bigNet', help='Feature network mode')


    parser.add_argument("--generate_texture",                                        action='store_true',                    help='Generate texture map.')
    parser.add_argument("--fg_loss_only",                                        action='store_true',                    help='Appy loss on the foreground only')


    parser.add_argument("--light_mlp_input", type=str, default='uv', help='Feature network mode')

    parser.add_argument("--supervise_img_mask_fg",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--w_img_donut_loss",            type=float,         default=1.0,                                       help='Set to adjust structural similarity loss weight')
    parser.add_argument("--w_img_FG_loss",            type=float,         default=1.0,                                       help='Set to adjust structural similarity loss weight')
    parser.add_argument("--use_inner_jelly",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--render_with_tf_function",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--load_best_model",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--use_mse_supervision",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--test_mode", type=str, default='train', help='whether to test on train or test frames')
    parser.add_argument("--shuffle_buffer",            type=int,           default=1000,                                                help='Frequency of console printout and metric logging')
    parser.add_argument("--inference",                                        action='store_true',                    help='Appy loss on the foreground only')
    parser.add_argument("--bullet_time",                                        action='store_true',                    help='Render bullet time (fixed frame) free viewpoint video')
    parser.add_argument("--in_jelly_offset",            type=float,         default=30.0,                                       help='Inner jelly offset')
    parser.add_argument("--out_jelly_offset",            type=float,         default=40.0,                                       help='Outer jelly offset')
    parser.add_argument("--add_input",                                        action='store_true',                    help='Whether to add the input to the intermediate feature')
    parser.add_argument("--no_skips",                                        action='store_true',                    help='Whether to add the input to the intermediate feature')
    parser.add_argument("--deep_skip_unet",                                        action='store_true',                    help='Whether to add the input to the intermediate feature')
    parser.add_argument("--use_grad_mask",                                        action='store_true',                    help='use gradient masking')


    parser.add_argument("--originalResolutionU",        type=int,           default=1024,                                               help='Training U resolution')
    parser.add_argument("--originalResolutionV",        type=int,           default=1024,                                               help='Training U resolution')

    return parser
