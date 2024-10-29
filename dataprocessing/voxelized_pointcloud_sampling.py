from scipy.spatial import cKDTree as KDTree
import numpy as np
import os
import traceback
import igl
import open3d as o3d

kdtree, grid_points, cfg = None, None, None
def voxelized_pointcloud_sampling(path):
    try:

        out_path = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(cfg.input_res, cfg.num_points)


        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            return

        mesh = o3d.io.read_triangle_mesh(input_file) # 三角网格进行高级操作如点云采样,采用open3d的read,返回一个 open3d.geometry.TriangleMesh 对象
        pcl = mesh.sample_points_poisson_disk(cfg.num_points) # 泊松盘采样（Poisson disk sampling）方法对网格模型进行点云采样，生成 cfg.num_points 个点
        point_cloud = np.asarray(pcl.points) # point_cloud 是一个 N x 3 的 numpy 数组，每一行代表一个采样点的坐标
        pcf=np.array([[0,0,0]]) # pcf 是一个占位符，表示三维点云文件的面数据（这里没有定义任何面）
        igl.write_obj(out_path + '/pc.obj', point_cloud, pcf) # 保存时仅包含点云数据，而不包含拓扑信息
        
        occupancies = np.zeros(len(grid_points), dtype=np.int8) # 设置占用数组

        _, idx = kdtree.query(point_cloud) # 使用 KD 树查询每个点云点，找到它在 grid_points 中的最近网格点的索引 idx
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies) # 二进制数组 occupancies 压缩成字节，以减少文件大小

        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = cfg.bb_min, bb_max = cfg.bb_max, res = cfg.input_res)
        print('Finished: {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def init(cfg_param):
    global kdtree, grid_points, cfg
    cfg = cfg_param
    grid_points = create_grid_points_from_bounds(cfg.bb_min, cfg.bb_max, cfg.input_res) # BOX的范围，以及resolution分辨率
    kdtree = KDTree(grid_points) # 多维空间中进行高效的最近邻查询、范围搜索等操作

def create_grid_points_from_bounds(minimun, maximum, res): # 将三维box由空间中的点来表示, 返回最终所有点的坐标
    x = np.linspace(minimun, maximum, res) # 在 minimun 和 maximum 之间生成 res 个均匀分布的点
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij') # 将 x 的值扩展到三维坐标系中，使每个坐标组合都对应一个点
    X = X.reshape((np.prod(X.shape),)) # 将X展开为一维数组，便于之后合并坐标
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z)) # X,Y,Z分别代表一列,合并为最后的所有坐标点，每一行为(x,y,z)
    del X, Y, Z, x
    return points_list