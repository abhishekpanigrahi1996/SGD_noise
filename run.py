import os
import subprocess
import itertools
import time
import numpy as np

# folder to save
base_path = '/datadrive/results_sgd_tail_index'

if not os.path.exists(base_path):
    os.makedirs(base_path)

# server setup
launcher = "srun --nodes=1 --gres=gpu:1 --time=40:00:00 "# THIS IS AN EXAMPLE!!!

# experimental setup
width = [512]# [256, 512, 1024]#4, 8, 16, 32, 64, 128, 256, 512, 1024]
depth = [3]#2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds = list(range(1))
dataset = ['cifar10']#['cifar10']#['mnist']
loss = ['NLL']#'linear_hinge']
model = ['alexnet']#, 'Resnet', 'VGG16']
#add_gaussian_noise = 0
gaussian_variance  = [0]#, 1e-1, 1e-2]
learning_rate      = np.power(2.0, -10)
#mnist_dir = '/home/msridnnadmin/MNIST/mnist_challenge/MNIST_data/'
subset = -1#8192
batch_size_train = [256]#[32, 1024, 8192]
cifar_dir = '/datadrive/cifar10_data/'
activation = ['relu']
#noise_batch_size = []


grid = itertools.product(width, depth, seeds, dataset, loss, gaussian_variance, batch_size_train, model, activation)

processes = []
for w, dep, s, d, l, gauss, bt, m, act in grid:
    if m not in ['fc'] and dep == 9:
       continue  
    if gauss > 0 and bt != 8192:
       continue

    #load_dir = base_path + '/{}_{:04d}_{:02d}_{}_{}_{}_{}_subset{}_lr{}_GN{}_NV{}_bnfalse'.format(dep, w, s, d, l, m, batch_size_train, subset, learning_rate, add_gaussian_noise, gaussain_variance)
    #load_dir = base_path + '/{}_{:04d}_{:02d}_{}_{}_{}_{}_subset{}_lr{}_gauss{}_noisebatch{}_bnTrue_ExtensiveGradient_defaultinit'.format(dep, w, s, d, l, m, bt, subset, learning_rate, gauss, nbst)
    save_dir = base_path + '/{}_{:04d}_{:02d}_{}_{}_{}_{}_subset{}_lr{}_gauss{}_bnFalse'.format(dep, w, s, d, l, m, bt, subset, learning_rate, gauss)
    #if os.path.exists(save_dir):
        # folder created only at the end when all is done!
    #    print('folder already exists, quitting')
    #    continue

    cmd =  ''#launcher + ' '
    cmd += 'python main.py '
    cmd += '--save_dir {} '.format(save_dir)
    cmd += '--width {} '.format(w)
    cmd += '--depth {} '.format(dep)
    cmd += '--seed {} '.format(s)
    cmd += '--dataset {} '.format(d)
    cmd += '--model {} '.format(m)
    cmd += '--lr {} '.format(str(learning_rate))
    #cmd += '--lr_schedule '
    cmd += '--iterations {} '.format(100)
    cmd += '--path {} '.format(cifar_dir)
    cmd += '--batch_size_train {} '.format(bt)
    cmd += '--batch_size_eval {} '.format(4096)
    cmd += '--noise_batch_size_train {} '.format(1)
    cmd += '--eval_freq {} '.format(1) 
    if gauss != 0:   
        cmd += '--add_gaussian_noise {} '.format(1)
    cmd += '--gaussian_variance {} '.format(gauss)
    cmd += '--subset {} '.format(subset) 
    cmd += '--act {} '.format(act)
    cmd += '--grad_sample_size {} '.format(25000)
    #cmd += '--batch_norm {} '.format(1) 
    #cmd += '--load_dir {} '.format(load_dir) 
    # cmd += '--print_freq {} '.format(1), # dbg
    # cmd += '--verbose '

    #print(cmd)
    
    #f = open(save_dir + '.log', 'w') 
    #processes.append(subprocess.Popen(cmd.split()))#.wait()
    os.system(cmd)
    
#for process in processes:
#    process.wait()
