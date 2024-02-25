
iteration           = 1
calib               = cameras

### Vlad ###
basedir             = /CT/ykwon/work/deepcharacters/Projects/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogArticulatedNerf
datadir             = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew
meshfilepath        = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/vlad.obj
charFilePath        = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/vlad.character

mesh_seq_path        = /CT/ashwath/work/DDC_DATA/VladNew/results_full_chamfer/tensorboardLogDeepDynamicCharacters/3709x3756x3760x107x109x132/snapshot_iter_359999/140_test_0_18900/final.meshes

imagedir            = /scratch/camstore/marc_tmp/mhaberma/Vlad/trainingNerf/rgb_1285_940
train_start_frame   = 100
train_end_frame     = 18800
train_cameras       = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
validation_cameras  = [14]
test_cameras        = [7,14,18,27,40]


resolutionU         = 1285
resolutionV         = 940

use_viewdirs        = False
depth               = True
positional_pose     = True
multiProcessing     = True
dilate_mask         = True
use_gt_tex          = False
random_seed         = 99


chunk = 4096
N_rand = 4096
N_samples = 1
num_views_per_batch = 1


netwidth = 256


no_reload           = False

#################### path

runner_module = Libs.Runner.model_main.run_nerf_depthpeel
runner_path = Libs/Runner/model_main/run_nerf_depthpeel.py

encoder_module = Libs.Networks.model_main.UNet
encoder_path = Libs/Networks/model_main/UNet.py



lrate_decay = 800
lrate = 2e-4
i_img = 200
i_weights = 10000
N_batch = 1









expname = vlad_l1 # first train with l1-only loss

# later train together with vgg loss vlad_l1_vgg

shuffle_buffer = 1000
use_perceptual_loss = False # Set to True when using perceptual supervision
w_perceptual_loss = 0.00001
fg_loss_only = True
textureFeatureSize = 32
w_img_FG_loss = 1.0
supervise_img_mask_fg = False
use_inner_jelly = False
use_mse_supervision = False



netdepth = 7
net_mode = bigNet
light_mlp_input = uv