#!/usr/bin/env python
import time
import os
import sys
import argparse
import numpy as np
from robot import Robot
from simulation import vrep
import cv2
import pygame


def capture_image(robot):
    path = "frame.jpg"
    if os.path.exists(path):
        os.remove(path)
    ig, _ = robot.get_camera_data()
    cv2.imwrite(path, ig)


def move(robot, increment):
    sim_ret, UR5_target_position = vrep.simxGetObjectPosition(robot.sim_client, robot.UR5_target_handle, -1,
                                                              vrep.simx_opmode_blocking)
    vrep.simxSetObjectPosition(robot.sim_client, robot.UR5_target_handle, -1,
                               UR5_target_position + increment,
                               vrep.simx_opmode_blocking)
    print("Position:", UR5_target_position + increment)


def rotate(robot, increment_angle):
    # Compute gripper orientation and rotation increments
    sim_ret, barrett_rot_handle = vrep.simxGetObjectHandle(robot.sim_client, './jointA_0',
                                                           vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
    sim_ret, gripper_rot_position = vrep.simxGetJointPosition(robot.sim_client, barrett_rot_handle,
                                                              vrep.simx_opmode_blocking)

    sim_ret, barrett_rot_handle_2 = vrep.simxGetObjectHandle(robot.sim_client, './jointA_2',
                                                             vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
    sim_ret, gripper_rot_position_2 = vrep.simxGetJointPosition(robot.sim_client, barrett_rot_handle,
                                                                vrep.simx_opmode_blocking)

    vrep.simxSetJointTargetPosition(robot.sim_client, barrett_rot_handle, gripper_rot_position + increment_angle,
                                    vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(robot.sim_client, barrett_rot_handle_2, gripper_rot_position_2 + increment_angle,
                                    vrep.simx_opmode_blocking)


def rotate_finger(robot, increment_angle):
    # Compute gripper orientation and rotation increments
    sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(robot.sim_client, robot.UR5_target_handle, -1,
                                                                 vrep.simx_opmode_blocking)

    vrep.simxSetObjectOrientation(robot.sim_client, robot.UR5_target_handle, -1,
                                  (np.pi / 2, gripper_orientation[1] + increment_angle, np.pi / 2),
                                  vrep.simx_opmode_blocking)
    print("Orientation:", np.floor((gripper_orientation[1] + increment_angle) / np.pi * 180), "degree")


def grasp(robot, evaluation=False):
    # Ensure gripper is open
    robot.open_gripper()

    sim_ret, position = vrep.simxGetObjectPosition(robot.sim_client, robot.UR5_target_handle, -1,
                                                   vrep.simx_opmode_blocking)

    # Close gripper to grasp target
    gripper_full_closed = robot.close_gripper()
    # Move gripper up
    grasp_location_margin = 0.2
    location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
    robot.move_to(location_above_grasp_target, None)

    # Check if pick up is successful
    object_positions = np.asarray(robot.get_obj_positions())
    object_positions = object_positions[:, 2]
    grasped_object_ind = np.argmax(object_positions)
    grasped_object_height = np.max(object_positions)
    grasped_object_handle = robot.object_handles[grasped_object_ind]
    grasp_success = grasped_object_height > 0.2
    # Move the grasped object elsewhere
    if grasp_success:
        vrep.simxSetObjectPosition(robot.sim_client, grasped_object_handle, -1,
                                   (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1),
                                   vrep.simx_opmode_blocking)

    robot.open_gripper()
    return grasp_success


# control robot by keyboard
def main(args):
    # --------------- Setup options ---------------
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir)
    num_obj = args.num_obj
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                       0.6]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(True, obj_mesh_dir, num_obj, workspace_limits, gripper_type='barrett')
    # Ensure gripper is open
    robot.open_gripper()

    # --------------- Setup pygame---------------
    pygame.init()

    display = pygame.display.set_mode((200, 30))
    pygame.display.set_caption("Vrep manual pick-up")

    # repeat event, set KEYDOWN multiple times (delay,freq)
    pygame.key.set_repeat(100, 50)

    # step per press
    step = 0.005
    angle_increment = np.pi / 16
    finger_angle_increment = np.pi / 12

    while True:

        # # Image display
        # capture_image(robot)
        # ig = pygame.image.load("frame.jpg")
        # display.blit(ig, (0, 0))
        # pygame.display.update()

        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    sys.exit()

                # move along X
                elif event.key == pygame.K_d:
                    increment = np.array([step, 0, 0])
                    move(robot, increment)
                elif event.key == pygame.K_a:
                    increment = np.array([-step, 0, 0])
                    move(robot, increment)

                # move along Y
                elif event.key == pygame.K_w:
                    increment = np.array([0, step, 0])
                    move(robot, increment)
                elif event.key == pygame.K_s:
                    increment = np.array([0, -step, 0])
                    move(robot, increment)

                # move along Z
                elif event.key == pygame.K_UP:
                    increment = np.array([0, 0, step])
                    move(robot, increment)
                elif event.key == pygame.K_DOWN:
                    increment = np.array([0, 0, -step])
                    move(robot, increment)

                # rotate along Z (counter clock-wise)
                elif event.key == pygame.K_q:
                    rotate(robot, angle_increment)

                # rotate along Z (clock-wise)
                elif event.key == pygame.K_e:
                    rotate(robot, -angle_increment)

                # rotate Barrett fingers (increase angle)
                elif event.key == pygame.K_z:
                    rotate_finger(robot, finger_angle_increment)

                # rotate Barrett fingers (decrease angle)
                elif event.key == pygame.K_x:
                    rotate_finger(robot, -finger_angle_increment)

            # gasp
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    print("succeess:", grasp(robot))
                    print("Execute grasp action...")

            # restart
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    print("Restart Game")
                    robot.restart_sim()
                    robot.add_objects()
                    robot.open_gripper()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/Test',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=3,
                        help='number of objects to add to simulation')
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
