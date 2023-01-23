#!/usr/bin/env python
import time
import os
import argparse
import numpy as np
from robot import Robot
from simulation import vrep
import utils
from logger import Logger


def main(args):
    # --------------- Setup options ---------------
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir)
    num_obj = args.num_obj
    save_dir = args.save_dir
    data_size = args.data_size
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                       0.6]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

    # Initialize logger
    logger = Logger(continue_logging=False, logging_directory=save_dir)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(True, obj_mesh_dir, num_obj, workspace_limits, 'barrett')

    for i in range(data_size):
        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose,
                                                               workspace_limits, 0.002)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(i, color_img, depth_img, '0')
        logger.save_heightmaps(i, color_heightmap, valid_depth_heightmap, '0')

        robot.obj_mesh_ind = np.random.randint(0, len(robot.mesh_list), size=robot.num_obj)
        robot.restart_sim()
        robot.add_objects()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/train',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=1,
                        help='number of objects to add to simulation')
    parser.add_argument('--data_size', dest='data_size', type=int, action='store', default=50,
                        help='number of data size')
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='manual_label/train',
                        help='save directory')
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
