#!/usr/bin/env python
import time
import os
import argparse
import numpy as np
import cv2
import torch
from robot import Robot
from trainer import MultiQTrainer, get_prediction_vis
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

    # ------------- Test options -------------
    is_test = args.is_test
    model_snapshot = args.model_snapshot
    test_model_type = args.model_type

    # ------------- Algorithm options -------------
    network_type = args.network_type
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay  # Use prioritized experience replay?

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? It takes some time.

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-up system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits, gripper_type)

    # Initialize trainer
    if is_test:
        trainer = MultiQTrainer(test_model_type, future_reward_discount, True, model_snapshot, force_cpu)
    else:
        trainer = MultiQTrainer(network_type, future_reward_discount, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose,
                            robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of preloaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize change count
    no_change_count = [0, 0]

    # Quick share for nonlocal memory between threads
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_pix_ind': None,
                          'grasp_success': None}

    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # best_grasp_1_conf = np.max(grasp_1_predictions)
                # best_grasp_2_conf = np.max(grasp_2_predictions)

                # print('Primitive confidence scores: %f (grasp_1), %f (grasp_2)' % (
                #     best_grasp_1_conf, best_grasp_2_conf))
                # nonlocal_variables['primitive_action'] = 'grasp_2'
                # if best_grasp_1_conf > best_grasp_2_conf:
                #     nonlocal_variables['primitive_action'] = 'grasp_1'
                nonlocal_variables['primitive_action'] = 'grasp_2'

                # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                if nonlocal_variables['primitive_action'] == 'grasp_1':
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_1_predictions),
                                                                          grasp_1_predictions.shape)
                    predicted_value = np.max(grasp_1_predictions)
                elif nonlocal_variables['primitive_action'] == 'grasp_2':
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_2_predictions),
                                                                          grasp_2_predictions.shape)
                    predicted_value = np.max(grasp_2_predictions)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (
                    nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0],
                    nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(
                    nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'grasp_1':
                    trainer.executed_action_log.append(
                        [1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
                         nonlocal_variables['best_pix_ind'][2]])  # 1 - grasp_1
                elif nonlocal_variables['primitive_action'] == 'grasp_2':
                    trainer.executed_action_log.append(
                        [2, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
                         nonlocal_variables['best_pix_ind'][2]])  # 2 - grasp_2
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    grasp_1_pred_vis = get_prediction_vis(grasp_1_predictions, color_heightmap,
                                                          nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_1_pred_vis, 'grasp_1')
                    cv2.imwrite('visualization.grasp_1.png', grasp_1_pred_vis)
                    grasp_2_pred_vis = get_prediction_vis(grasp_2_predictions, color_heightmap,
                                                          nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_2_pred_vis, 'grasp_2')
                    cv2.imwrite('visualization.grasp_2.png', grasp_2_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['grasp_1_success'] = False
                nonlocal_variables['grasp_2_success'] = False
                change_detected = False

                if primitive_position[2] < 0.15:

                    # Execute primitive
                    if nonlocal_variables['primitive_action'] == 'grasp_1':
                        trainer.grasp_mode_count[0] += 1  # count mode1 occurrence
                        logger.writer.add_scalar('mode1', trainer.grasp_mode_count[0], trainer.iteration)
                        nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle,
                                                                          - np.pi / 6,
                                                                          workspace_limits)
                        print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    elif nonlocal_variables['primitive_action'] == 'grasp_2':
                        trainer.grasp_mode_count[1] += 1  # count mode2 occurrence
                        logger.writer.add_scalar('mode2', trainer.grasp_mode_count[1], trainer.iteration)
                        nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle,
                                                                          - np.pi / 2,
                                                                          workspace_limits)
                        print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                else:
                    print('Misidentify!')
                    robot.restart_sim()
                    robot.add_objects()

                nonlocal_variables['executing_action'] = False
            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
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
            grasp_1_predictions, grasp_2_predictions, state_feat = trainer.forward(color_heightmap,
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
                if prev_primitive_action == 'grasp_1':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp_2':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'grasp_1':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp_2':
                    no_change_count[1] += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_grasp_success, color_heightmap,
                                                                     valid_depth_heightmap)
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            grasp_type = 1 if nonlocal_variables['primitive_action'] == 'grasp_1' else 2

            if not is_test:
                # Backpropagate
                loss_value = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_best_pix_ind,
                                              label_value,
                                              grasp_type)
                logger.writer.add_scalar('loss', loss_value, trainer.iteration)

            # Do sampling for experience replay
            if experience_replay:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'grasp_1':
                    sample_primitive_action_id = 1
                    sample_reward_value = 0 if prev_reward_value == 1 else 1

                elif sample_primitive_action == 'grasp_2':
                    sample_primitive_action_id = 2
                    sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(
                    np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration, 0] == sample_reward_value,
                                   np.asarray(trainer.executed_action_log)[1:trainer.iteration,
                                   0] == sample_primitive_action_id))

                if sample_ind.size > 0:

                    # Find sample with the highest surprise value
                    sample_surprise_values = np.abs(
                        np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - np.asarray(trainer.label_value_log)[
                            sample_ind[:, 0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (
                        sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(
                        os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(
                        os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_grasp_1_predictions, sample_grasp_2_predictions, sample_state_feat = trainer.forward(
                            sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(
                        os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration + 1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(
                        os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration + 1)),
                        -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32) / 100000

                    # Get labels for sample and backpropagation
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_best_pix_ind,
                                     trainer.label_value_log[sample_iteration],
                                     trainer.executed_action_log[sample_iteration][0])

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'grasp_1':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_1_predictions)]
                        # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                    elif sample_primitive_action == 'grasp_2':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_2_predictions)]
                        # trainer.label_value_log[sample_iteration] = [new_sample_label_value]

                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            logger.save_backup_model(trainer.model, 'multiQ')
            if trainer.iteration % 100 == 0:
                logger.save_model(trainer.iteration, trainer.model, 'multiQ')
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
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robot to learn visual affordance with FCN')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/train_household',
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
    parser.add_argument('--network', dest='network_type', action='store',
                        default='student')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,
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

    # ----- Model Evaluation options ------
    parser.add_argument('--is_test', dest='is_test', action='store_true', default=False,
                        help='model evaluation')
    parser.add_argument('--model_snapshot', dest='model_snapshot', action='store')
    parser.add_argument('--model_type', dest='model_type', action='store')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
