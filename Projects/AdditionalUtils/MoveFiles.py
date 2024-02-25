import os

# move camera files

src_root = '/scratch/camstore/marc_tmp/mhaberma/Vlad/trainingNerf/depth_1_4112_3008/cameras'
dst_root = '/scratch/inf0/user/ykwon/Vlad/trainingNerf/depth_1_4112_3008/cameras'


n_cams = 101

for cam_idx in range(n_cams):

    src_dir = os.path.join(src_root,str(cam_idx))
    dst_dir = os.path.join(dst_root, str(cam_idx))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    frame = 100

    cmd = 'cp {}/f_{}.npz {}/f_{}.npz'.format(src_dir, frame, dst_dir, frame)
    os.system(cmd)
