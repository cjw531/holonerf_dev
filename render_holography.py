import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
    except RuntimeError as e:
        print(e)

import numpy as np
import imageio
import pprint
import matplotlib.pyplot as plt
import math

import run_nerf
from run_nerf import *
from load_llff import load_llff_data


basedir = './logs'
expname = 'polaroid_new'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf.config_parser()

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_100000.npy')))
print('loaded args')
print("spherify: ", args.spherify)
print("center_n_rotate: ", args.center_n_rotate)

images, poses, bds, render_poses, i_test, _ = load_llff_data(args.datadir, args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify, center_n_rotate=args.center_n_rotate)

H, W, focal = poses[0,:3,-1].astype(np.float32)
H = int(H)
W = int(W)
hwf = [H, W, focal]

images = images.astype(np.float32)
poses = poses.astype(np.float32)

if args.no_ndc:
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
else:
    near = 0.
    far = 1.

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

bds_dict = {
    'near' : tf.cast(near, tf.float32),
    'far' : tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)

render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
render_kwargs_fast['N_importance'] = 0

c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix

down = 4 # trade off resolution+aliasing for render speed to make this video faster
frames = []
for i, c2w in enumerate(render_poses):
    if i%8==0: print(i)
    test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w[:3,:4], **render_kwargs_fast)
    frames.append((255*np.clip(test[0],0,1)).astype(np.uint8))
    
print('done, saving')
f = basedir + '/' + expname + '/video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=8)
