import torch
import render
from scipy.spatial.transform import Rotation as R
import numpy as np
from pypcd import *
# import pcl.pcl_visualization
# import open3d as o3d
from open3d import *
from plyfile import *
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


r = R.from_euler('zxy', [10,80,4], degrees=True)
lidar_rotation = torch.tensor(r.as_matrix(), dtype=torch.float).cuda()


class attack_msf():
    def __init__(self):
        pass


    def loadPCL(self, PCL, flag=True):
        if flag:
            PCL = np.fromfile(PCL, dtype=np.float32)
            PCL = PCL.reshape((-1, 4))
        else:
            PCL = pypcd.PointCloud.from_path(PCL)
            PCL = np.array(tuple(PCL.pc_data.tolist()))
            PCL = np.delete(PCL, -1, axis=1)
        return PCL


    def load_pc_mesh(self, path):
        PCL_path = path
        # loading ray_direction and distance for the background pcd
        self.PCL = self.loadPCL(PCL_path, True)
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)


    def load_mesh(self, path, r, x_of=0, y_of=0):
        z_of = 0
        plydata = PlyData.read(path)
        x = torch.FloatTensor(plydata['vertex']['x']) * r
        y = torch.FloatTensor(plydata['vertex']['y']) * r
        z = torch.FloatTensor(plydata['vertex']['z']) * r
        self.object_v = torch.stack([x, y, z], dim=1).cuda()

        self.object_f = plydata['face'].data['vertex_indices']
        # self.object_f = plydata['face'].data['vertex_index']
        self.object_f =  torch.tensor(np.vstack(self.object_f)).cuda()



    def rendering_img(self):

        point_cloud_rendered = render.render(self.ray_direction, self.length, self.object_v, self.object_f, self.i_final)
        point_cloud_rendered = point_cloud_rendered.cpu()

        source_data = point_cloud_rendered[:, 0:3].reshape(-1, 3)  # 10000x3
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(source_data)
        # open3d.visualization.draw_geometries([point_cloud])
        open3d.io.write_point_cloud('../../data/msf-adv/test.ply', point_cloud)

        save_path1 = r"../../data/msf-adv/matplotlib-mesh.png"
        save_path2 = r"../../data/msf-adv/matplotlib-bg.png"
        # self.visualize(source_data[:61940, :], save_path1)
        # self.visualize(source_data[61940:, :], save_path2)



    def visualize(self, points, save_path):
        # filePath = './GEN_Ours_airplane_1631630214/out.npy'
        # points = np.load(filePath)  # (n, 3)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='y')
        print(points.shape)
        print(points[:, 0].shape, points[:, 1].shape, points[:, 2].shape)
        print(points)
        plt.savefig(save_path)
        plt.show()



def test():
    obj = attack_msf()
    pass

if __name__ == '__main__':

    test()
    path_object = r'../../data/msf-adv/object.ply'
    path_lidar_background = r'../../data/msf-adv/lidar.bin'

    obj = attack_msf()

    obj.load_mesh(path_object, 10.0)
    obj.load_pc_mesh(path_lidar_background)
    obj.rendering_img()

