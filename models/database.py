import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad



class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed, grid):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        self.data=torch.cat((points,speed),dim=1)
        self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)
        self.grid    = self.grid.to(device)


    def __getitem__(self, index):
        data = self.data[index]
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    try:
        points = np.load('{}/sampled_points.npy'.format(PATH)) #(x0,y0,z0,x1,y1,z1)
        speed = np.load('{}/speed.npy'.format(PATH)) #(s0,s1)
        occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
        input = np.reshape(occupancies, (128,)*3) # (128,128,128)体素占用网格，每个位置的值为 0（未占用）或 1（被占用）
        grid = np.array(input, dtype=np.float32) # 转为浮点数
        #print(tau.min())
    except ValueError:
        print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]
    
    print(points.shape,speed.shape)
    print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed,grid)
    #database = _numpy2dataset(XP,YP)
    return database





