import sys

sys.path.append('.') # 当前工作目录（即 .）添加到模块搜索路径中，引用模块时可以在本目录中搜索

from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu import sample_speed
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os

cfg = cfg_loader.get_config()
print(cfg.data_dir)
print(cfg.input_data_glob)

print('Finding raw files for preprocessing.')
paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
print(paths)
paths = sorted(paths)



num_cpus = mp.cpu_count()


def multiprocess(func):
	p = Pool(num_cpus)
	# 将paths中的每一给item分别传给function,使这些function作为多线程并行执行
	p.map(func, paths)
	# 关闭线程池,使之不能再加入新的线程
	p.close()
	# 等待线程池中的线程结束
	p.join()

# 将空间规范到[-0.5,0.5]之间
print('Start scaling.')
multiprocess(to_off)
# sampled start and goal pairs in robot c-space randomly,同时计算ground truth速度
print('Start speed sampling.')
for path in paths:
	sample_speed(path, cfg.num_samples, cfg.num_dim)
# 将工作空间障碍物点云转化为占用体素网格
print('Start voxelized pointcloud sampling.')
voxelized_pointcloud_sampling.init(cfg)
multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)


