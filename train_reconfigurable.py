#!/usr/bin/env python
import time
import os
import argparse
import numpy as np
import cv2
import torch
from robot import Robot
from trainer import HybridTrainer, get_prediction_vis
from logger import Logger
import utils
import threading
import warnings
warnings.filterwarnings('ignore')  # CAREFULL!!! Supress all warnings


def main(args):
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    obj_mesh_dir = os.path.abspath(
        args.obj_mesh_dir) if is_sim else None  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None  # Number of objects to add to simulation
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                           0.4]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        pass  # TODO realworld workspace

    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    gripper_type = 'barrett'

    # ------------- Algorithm options -------------
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay  # Use prioritized experience replay?

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-up system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits, gripper_type)

    # Initialize trainer
    trainer = HybridTrainer(future_reward_discount, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose,
                            robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of preloaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = 0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'best_pix_ind': None,
                          'best_gripper_config': None,
                          'grasp_success': None}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Determine whether grasping or pushing should be executed based on network predictions
                nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions),
                                                                      grasp_predictions.shape)
                nonlocal_variables['best_gripper_config'] = config_predictions[nonlocal_variables['best_pix_ind']]
                print('best_gripper_config', nonlocal_variables['best_gripper_config'])
                predicted_value = np.max(grasp_predictions)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: grasp at (%d, %d, %d)' % (nonlocal_variables['best_pix_ind'][0],
                                                         nonlocal_variables['best_pix_ind'][1],
                                                         nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(
                    nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                primitive_config = nonlocal_variables['best_gripper_config']

                # Save executed primitive
                trainer.executed_action_log.append(
                    [1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
                     nonlocal_variables['best_pix_ind'][2], primitive_config])  # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    config_pred_vis = get_prediction_vis(config_predictions, color_heightmap,
                                                                 nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, config_pred_vis, 'config')
                    cv2.imwrite('visualization.config.png', config_pred_vis)
                    grasp_pred_vis = get_prediction_vis(grasp_predictions, color_heightmap,
                                                                nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['grasp_success'] = False

                if primitive_position[2] < 0.2:
                    # Execute primitive
                    nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle,
                                                                      (np.pi / 2 * primitive_config) - np.pi / 2,
                                                                      workspace_limits)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                else:
                    print('Misidentify the objects!')
                    robot.restart_sim()
                    robot.add_objects()

                nonlocal_variables['executing_action'] = False
            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits,
                                                               heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count > 10):
            no_change_count = 0
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (
                    np.sum(stuff_count)))
                robot.restart_real()
            continue

        if not exit_called:
            # Run forward pass with network to get affordances
            st = time.time()  # recording start time
            config_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap,
                                                                                valid_depth_heightmap,
                                                                                is_volatile=True)
            et = time.time()  # recording end time
            print('Execution time of forward passing:', et - st, 'seconds')
            # Execute the best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                no_change_count = 0

            else:
                no_change_count += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_grasp_success, color_heightmap,
                                                                     valid_depth_heightmap)
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_best_pix_ind,
                             label_value, prev_best_config)

            # Save model snapshot
            logger.save_backup_model(trainer.model, 'reconfig')
            if trainer.iteration % 100 == 0:
                logger.save_model(trainer.iteration, trainer.model, 'reconfig')
                if trainer.use_cuda:
                    trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_best_config = nonlocal_variables['best_gripper_config']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robot to learn visual affordance with FCN')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/train',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=3,
                        help='number of objects to add to simulation')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=True,
                        help='use prioritized experience replay?')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
