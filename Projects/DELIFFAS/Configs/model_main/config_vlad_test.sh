expname = vlad_l1_vgg

load_best_model     = False
ft_iteration        = None #

basedir             = /CT/ykwon/work/deepcharacters/Projects/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogArticulatedNerf
datadir             = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew
meshfilepath        = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/vlad.obj
charFilePath        = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/vlad.character

################# free viewpoint rendering #################

### train ####
#mesh_seq_path        = # Mesh sequence file (.meshes) for novel view task

#train_start_frame   = 100
#train_end_frame     = 17000
#test_mode           = novel_view_task_freeview

### test ####
#mesh_seq_path = # Mesh sequence file (.meshes) for novel pose task


#train_start_frame   = 100
#train_end_frame     = 7300
#test_mode           = novel_pose_task_freeview

#texdir              = /scratch/inf0/user/mhaberma/Vlad/trainingNerf_poseOnly/tex_0/tex/ #/scratch/inf0/user/ykwon/Vlad/trainingNerf
#depthdir            = /scratch/inf0/user/ykwon/Vlad/trainingNerf
#calibdir            = /CT/ykwon/work/deepcharacters/Projects/DeepDynamicCharacters/Dataset/calibrations
#imagedir            = /scratch/camstore/marc_tmp/mhaberma/Vlad/trainingNerf/rgb_1285_940
#fgmaskdir           = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/training/foregroundSegmentation
#calib               = rotation #rotation_x #rotation_steps_100 #rotation #zooooom ##
#moving_cam          = True # when free-viewpoint rendering, should be set as True
#render_only         = True
#bullet_time         = False



################# quantitative evaluation (novel view) #################
#mesh_seq_path        = # Mesh sequence file (.meshes) for novel view task

#texdir              = /scratch/inf0/user/ykwon/Vlad/trainingNerf
#depthdir            = /scratch/inf0/user/ykwon/Vlad/trainingNerf
#imagedir            = /scratch/camstore/marc_tmp/mhaberma/Vlad/trainingNerf/rgb_1285_940
#fgmaskdir           = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/training/foregroundSegmentation
#calib               = cameras
#test_cameras        = [7,18,27,40]
#train_start_frame   = 100
#train_end_frame     = 17000
#quantitative_eval   = False
#moving_cam          = False
#render_only         = True
#test_mode           = eval_novel_view

################# quantitative evaluation (novel_pose) #################
mesh_seq_path = # Mesh sequence file (.meshes) for novel pose task

texdir              = /scratch/inf0/user/ykwon/Vlad/trainingNerf
depthdir            = /scratch/inf0/user/ykwon/Vlad/trainingNerf
imagedir            = /scratch/camstore/marc_tmp/mhaberma/Vlad/trainingNerf/rgb_1285_940
fgmaskdir           = /HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/training/foregroundSegmentation
calib               = cameras

test_cameras        = [7,18,27,40]
train_start_frame   = 100
train_end_frame     = 7300
quantitative_eval   = False
moving_cam          = False
render_only         = True
test_mode           = eval_novel_pose

use_gt_tex          = False
create_pc           = False

iteration           = 1
resolutionU         = 1285
resolutionV         = 940
use_viewdirs        = False
depth               = True
positional_pose     = True
perturb             = 0.0
raw_noise_std       = 0.0

netdepth = 7
netwidth = 256
textureFeatureSize = 32
num_views_per_batch = 1
N_batch = 1

net_mode = bigNet
light_mlp_input = uv


##### when evaluating speed,
# in "config_vlad_test.sh" render_with_tf_function=True, inference=False
# in "run.py"
#1.from Utils.model_main.load_data_measure_eval_time import \
#    NerfDataloader
#2.in def batchify_rays,
# chunk = whereFG.shape[1]

##### when not evaluating speed,
# in "config_vlad_test.sh" render_with_tf_function=False, inference=True
# in "run.py"
#1. from Utils.model_main.load_data import \
#    NerfDataloader
#2.in def batchify_rays,
#        if num_rays[0][0] > 50000:
#            chunk = 40960
#        else:
#            chunk = whereFG.shape[1]

render_with_tf_function = False # True (evaluating speed) # False (not evaluating speed)
use_inner_jelly = False
inference = True # whether fixing missing pixel should be enabled


runner_module = Libs.Runner.model_main.run
runner_path = Libs/Runner/model_main/run.py

# if you want to evaluate speed, use run_eval_speed for the runner
#runner_module = Libs.Runner.model_main.run_eval_speed
#runner_path = Libs/Runner/model_main/run_eval_speed.py

encoder_module = Libs.Networks.model_main.UNet
encoder_path = Libs/Networks/model_main/UNet.py
