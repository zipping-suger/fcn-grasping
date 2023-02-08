import time
import utils
import cv2
import os
import numpy as np
from robot_test import Robot
from trainer import MultiQTrainer, get_prediction_vis
import torch
import warnings
warnings.filterwarnings('ignore')  # CAREFULL!!! Supress all warnings

if __name__ == '__main__':
    # Parameters
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    is_sim = True
    obj_mesh_dir = '../objects/train'
    obj_mesh_dir = os.path.abspath(obj_mesh_dir)
    num_obj = 3
    gripper_type = 'barrett'
    heightmap_resolution = 0.002

    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits, gripper_type)
    robot.restart_sim()

    j = 1

    # Get latest RGB-D image
    color_img, depth_img = robot.get_in_hand_camera_data()
    depth_img = depth_img * robot.in_hand_cam_depth_scale  # Apply depth scale from calibration

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.in_hand_cam_intrinsics,
                                                           robot.in_hand_cam_pose, workspace_limits,
                                                           heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    trainer = MultiQTrainer('student', future_reward_discount=0, load_snapshot=False, snapshot_file=None,
                            force_cpu=False)

    snapshot_file = '../logs/student_household/models/snapshot-004400.multiQ.pth'

    trainer.model.load_state_dict(torch.load(snapshot_file))
    print('Pre-trained model snapshot loaded from: %s' % snapshot_file)

    grasp_predictions_1, grasp_predictions_2, state_feat = trainer.forward(color_heightmap,
                                                                           valid_depth_heightmap,
                                                                           is_volatile=True)
    best_pix_ind_1 = np.unravel_index(np.argmax(grasp_predictions_1), grasp_predictions_1.shape)
    best_pix_ind_2 = np.unravel_index(np.argmax(grasp_predictions_2), grasp_predictions_2.shape)

    graps_pred_1_vis = get_prediction_vis(grasp_predictions_1, color_heightmap, best_pix_ind_1)
    cv2.imwrite(os.path.join('%06d.0.grasp_1.png' % j), graps_pred_1_vis)
    grasp_pred_2_vis = get_prediction_vis(grasp_predictions_2, color_heightmap, best_pix_ind_2)
    cv2.imwrite(os.path.join('%06d.0.grasp_2.png' % j), grasp_pred_2_vis)

