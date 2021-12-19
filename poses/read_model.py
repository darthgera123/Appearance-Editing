import numpy as np
import os
import sys
import imageio
import skimage.transform

import poses.colmap_read_model as read_model

def load_colmap_data(realdir, sub_dir):
    
    camerasfile = os.path.join(realdir, 'dense/0/%s/cameras.bin' % sub_dir)
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'dense/0/%s/images.bin' % sub_dir)
    imdata = read_model.read_images_binary(imagesfile)
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    names = [imdata[k].name for k in imdata]
    camera_id =[imdata[k].camera_id for k in imdata]
    img_id = [imdata[k].id for k in imdata] 
    # name2camera = dict(zip(names,camera_id))
    name2camera = dict(zip(names,img_id))
    # print(name2camera)
    
    # print(dict(zip(names,camera_id)))
    # print( 'Images #', len(names))
    perm = np.argsort(names)
    cam2pose = {}
    for i,k in enumerate(imdata):
        im = imdata[k]
        # cam2pose[im.camera_id]=i
        cam2pose[im.id] = i
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # print()
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'dense/0/%s/points3D.bin' % sub_dir)
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    # poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    # poses = np.concatenate([-poses[:, 1:2, :], poses[:, 0:1, :], poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    # x,y,z axis we switched and took negative. This works for landscape mode, rotate left for portrait mode
    poses = np.concatenate([-poses[:, 0:1, :], -poses[:, 1:2, :], poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, name2camera,cam2pose
    
def camera_pose(path,img_name,sub_dir):
    poses,pts3d,perm,name2camera,cam2pose = load_colmap_data(path,sub_dir)
    camera_ind = name2camera[img_name]
    pose_ind = cam2pose[camera_ind]
    # ind = img_name
    # print(name2camera,cam2pose)
    # print(img_name,camera_ind,pose_ind)
    # print(poses.shape)
    pose = poses[:,:,pose_ind]
    width,height,focal = pose[:,-1]
    P = pose[:,:-1]
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    P = np.vstack((P,bottom))
    return P.flatten(), focal, width, height

if __name__ == '__main__':
    path = '/media/aakash/wd1/DATASETS/BUDDHA/colmap_output'
    # path = '/media/aakash/wd1/DATASETS/PINECONE/dense/0'
	# poses,pts3d,perm = load_colmap_data(path)
	# poses is 3*5*images, pts_3d are points, perm are images.
	# camera pos will be retrieved from poses
	# 3*5, last column is image height,width,focal length
	# refer here on how to get camera pos https://mathoverflow.net/questions/68385/calculate-camera-position-from-3x4-projection-matrix

	# pose = poses[:,:,60]
	# width,height,focal = pose[:,-1]
	# P = pose[:,:-1]
	# bottom = np.array([0,0,0,1.]).reshape([1,4])
	# P = np.vstack((P,bottom))
	# print(P.shape)
	# print(P.flatten())   
    p, focal_length, og_width, og_height = camera_pose(path,'0000000.png','sparse')
    pose = ','.join([str(elem) for elem in p])
    estimated_f = np.sqrt( pow(6.4, 2) + pow(4.8, 2) ) *\
     focal_length / np.sqrt( pow(og_width, 2) + pow(og_height, 2) )
    focal_length = estimated_f * 34.6 / 6.4  
    print(p)
    # print(p)
    print(focal_length)
    
	
