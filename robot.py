import socket
import struct
import time
import os
import numpy as np
import utils
from simulation import vrep


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits, gripper_type):
        self.is_sim = is_sim
        self.workspace_limits = workspace_limits
        self.gripper_type = gripper_type

        # If in simulation...
        if self.is_sim:
            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                           [89.0, 161.0, 79.0],  # green
                                           [156, 117, 95],  # brown
                                           [242, 142, 43],  # orange
                                           [237.0, 201.0, 72.0],  # yellow
                                           [186, 176, 172],  # gray
                                           [255.0, 87.0, 89.0],  # red
                                           [176, 122, 161],  # purple
                                           [118, 183, 178],  # cyan
                                           [255, 157, 167]]) / 255.0  # pink

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Connect to simulator
            vrep.simxFinish(-1)  # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # Add objects to simulation environment
            self.add_objects()

        # If in real-settings...
        else:
            # Connect to robot client
            self.tcp_socket = None
            self.tcp_host_ip = '192.168.1.10'
            self.tcp_port = 3002

            # Default home joint configuration
            # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
            self.home_joint_config = [(45 / 360.0) * 2 * np.pi, -(84.2 / 360.0) * 2 * np.pi,
                                      (112.8 / 360.0) * 2 * np.pi, -(119.7 / 360.0) * 2 * np.pi,
                                      -(90.0 / 360.0) * 2 * np.pi, 0.0]

            # Default joint speed configuration
            self.joint_acc = 1.4  # Safe: 1.4     # default = 8
            self.joint_vel = 1.05  # Safe: 1.05    # default = 3

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.2  # Safe: 0.5    # default = 1.2
            self.tool_vel = 0.25  # Safe: 0.2   # default = 0.25

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

            # Move robot to home pose
            # self.close_gripper()
            self.go_home()

            # Fetch RGB-D data from RealSense camera
            # from real.camera import Camera
            # self.camera = Camera()
            # self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            # self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            # self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',
                                                            vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

        sim_ret, self.in_hand_cam_handle = vrep.simxGetObjectHandle(self.sim_client, '/UR5/Vision_in_hand',
                                                                   vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, in_hand_cam_position = vrep.simxGetObjectPosition(self.sim_client, self.in_hand_cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, in_hand_cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.in_hand_cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        in_hand_cam_trans = np.eye(4, 4)
        in_hand_cam_trans[0:3, 3] = np.asarray(in_hand_cam_position)
        in_hand_cam_orientation = [-in_hand_cam_orientation[0], -in_hand_cam_orientation[1], -in_hand_cam_orientation[2]]
        in_hand_cam_rotm = np.eye(4, 4)
        in_hand_cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(in_hand_cam_orientation))
        self.in_hand_cam_pose = np.dot(in_hand_cam_trans, in_hand_cam_rotm)  # Compute rigid transformation representating camera pose
        self.in_hand_cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.in_hand_cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_in_hand_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)  # random objects sampling

        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                     self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                     self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1],
                            self.obj_mesh_color[object_idx][2]]
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                                  'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'importShape',
                                                                                                  [0, 0, 255, 0],
                                                                                                  object_position + object_orientation + object_color,
                                                                                                  [curr_mesh_file,
                                                                                                   curr_shape_name],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            time.sleep(1)
        self.prev_obj_positions = []
        self.obj_positions = []

    def restart_sim(self):
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3),
                                   vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)

    def restart_real(self):
        pass  # TODO

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        sim_ok = self.workspace_limits[0][0] - 0.1 < gripper_position[0] < self.workspace_limits[0][1] + 0.1 and \
                 self.workspace_limits[1][0] - 0.1 < gripper_position[1] < self.workspace_limits[1][1] + 0.1 and \
                 self.workspace_limits[2][0] < gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1,
                                                                        vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:
            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation,
                                          vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0,
                                                                           vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                    vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            pass  # TODO

        return color_img, depth_img

    def get_in_hand_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.in_hand_cam_handle, 0,
                                                                           vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.in_hand_cam_handle,
                                                                                    vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            pass  # TODO

        return color_img, depth_img

    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        robot_message_type = data_bytes[4]
        assert (robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx + 4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 8):(byte_idx + 16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data': parse_joint_data, 'cartesian_info': parse_cartesian_info,
                           'tool_data': parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def close_gripper(self):
        if self.is_sim:

            if self.gripper_type == "RG2":
                gripper_motor_velocity = -0.5
                gripper_motor_force = 100
                sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                                       vrep.simx_opmode_blocking)
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                            vrep.simx_opmode_blocking)
                vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                                vrep.simx_opmode_blocking)
                gripper_fully_closed = False
                while gripper_joint_position > -0.045:  # Block until gripper is fully closed
                    sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                                    vrep.simx_opmode_blocking)
                    # print(gripper_joint_position)
                    if new_gripper_joint_position >= gripper_joint_position:
                        return gripper_fully_closed
                    gripper_joint_position = new_gripper_joint_position
                gripper_fully_closed = True

            elif self.gripper_type == "barrett":
                gripper_motor_force = 100
                gripper_motor_velocity = 0.2

                sim_ret, barrett_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, './openCloseJoint',
                                                                           vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, barrett_gripper_handle,
                                                                            vrep.simx_opmode_blocking)

                sim_ret, barrett_gripper_handle_0 = vrep.simxGetObjectHandle(self.sim_client, './openCloseJoint0',
                                                                             vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
                sim_ret, gripper_joint_position_0 = vrep.simxGetJointPosition(self.sim_client, barrett_gripper_handle,
                                                                              vrep.simx_opmode_blocking)

                vrep.simxSetJointForce(self.sim_client, barrett_gripper_handle, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, barrett_gripper_handle, -gripper_motor_velocity,
                                                vrep.simx_opmode_blocking)

                vrep.simxSetJointForce(self.sim_client, barrett_gripper_handle_0, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, barrett_gripper_handle_0, -gripper_motor_velocity / 4,
                                                vrep.simx_opmode_blocking)

                gripper_fully_closed = False
                while gripper_joint_position > -0.15 or gripper_joint_position_0 > -0.19:  # Block until gripper is fully closed
                    sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client,
                                                                                    barrett_gripper_handle,
                                                                                    vrep.simx_opmode_blocking)
                    sim_ret, new_gripper_joint_position_0 = vrep.simxGetJointPosition(self.sim_client,
                                                                                      barrett_gripper_handle_0,
                                                                                      vrep.simx_opmode_blocking)
                    if new_gripper_joint_position >= gripper_joint_position and new_gripper_joint_position_0 >= gripper_joint_position_0:
                        return gripper_fully_closed
                    gripper_joint_position = new_gripper_joint_position
                    gripper_joint_position_0 = new_gripper_joint_position_0
                gripper_fully_closed = True

        else:
            pass  # TODO
        return gripper_fully_closed

    def open_gripper(self):
        if self.is_sim:
            if self.gripper_type == "RG2":
                gripper_motor_velocity = 0.5
                gripper_motor_force = 20
                sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                                       vrep.simx_opmode_blocking)
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                            vrep.simx_opmode_blocking)
                vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                                vrep.simx_opmode_blocking)
                while gripper_joint_position < 0.03:  # Block until gripper is fully open
                    sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                                vrep.simx_opmode_blocking)

            elif self.gripper_type == "barrett":
                gripper_motor_velocity = 0.15
                gripper_motor_force = 200
                sim_ret, barrett_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, './openCloseJoint',
                                                                           vrep.simx_opmode_blocking)
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, barrett_gripper_handle,
                                                                            vrep.simx_opmode_blocking)
                sim_ret, barrett_gripper_handle_0 = vrep.simxGetObjectHandle(self.sim_client, './openCloseJoint0',
                                                                             vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
                sim_ret, gripper_joint_position_0 = vrep.simxGetJointPosition(self.sim_client, barrett_gripper_handle,
                                                                              vrep.simx_opmode_blocking)

                vrep.simxSetJointForce(self.sim_client, barrett_gripper_handle, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, barrett_gripper_handle, gripper_motor_velocity,
                                                vrep.simx_opmode_blocking)
                vrep.simxSetJointForce(self.sim_client, barrett_gripper_handle_0, gripper_motor_force,
                                       vrep.simx_opmode_blocking)
                vrep.simxSetJointTargetVelocity(self.sim_client, barrett_gripper_handle_0, gripper_motor_velocity,
                                                vrep.simx_opmode_blocking)

                while gripper_joint_position < -0.1 and gripper_joint_position_0 < -0.1:  # Block until gripper is fully open
                    sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, barrett_gripper_handle,
                                                                                vrep.simx_opmode_blocking)
                    sim_ret, gripper_joint_position_0 = vrep.simxGetJointPosition(self.sim_client,
                                                                                  barrett_gripper_handle,
                                                                                  vrep.simx_opmode_blocking)

        else:
            pass  # TODO

    def rotate_fingers(self, theta):

        if self.is_sim:
            assert self.gripper_type == "barrett"
            sim_ret, barrett_rot_handle = vrep.simxGetObjectHandle(self.sim_client, './jointA_0',
                                                                   vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
            sim_ret, gripper_rot_position = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
                                                                      vrep.simx_opmode_blocking)

            sim_ret, barrett_rot_handle_2 = vrep.simxGetObjectHandle(self.sim_client, './jointA_2',
                                                                     vrep.simx_opmode_blocking)  # If not connected, sim_ret = 8
            sim_ret, gripper_rot_position_2 = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
                                                                        vrep.simx_opmode_blocking)

            vrep.simxSetJointTargetPosition(self.sim_client, barrett_rot_handle, theta, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetPosition(self.sim_client, barrett_rot_handle_2, theta, vrep.simx_opmode_blocking)

            sim_ret, gripper_rot_position = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
                                                                      vrep.simx_opmode_blocking)
            sim_ret, gripper_rot_position_2 = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
                                                                        vrep.simx_opmode_blocking)

            # while abs(gripper_rot_position - theta) > 0.001 and abs(
            #         gripper_rot_position_2 - theta) > 0.001:
            #     sim_ret, gripper_rot_position = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
            #                                                               vrep.simx_opmode_blocking)
            #     sim_ret, gripper_rot_position_2 = vrep.simxGetJointPosition(self.sim_client, barrett_rot_handle,
            #                                                                 vrep.simx_opmode_blocking)

    def get_state(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        while True:
            state_data = self.tcp_socket.recv(2048)
            if state_data[4] == 16:
                self.tcp_socket.close()
                return state_data

    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)

            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                    UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1],
                    UR5_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                          vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]),
                                       vrep.simx_opmode_blocking)

        else:

            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
                tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1],
                tool_orientation[2], self.tool_acc, self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            # Block until robot reaches target tool position
            tcp_state_data = self.get_state()
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all(
                    [np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
                # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
                tcp_state_data = self.get_state()
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time.sleep(0.01)
            self.tcp_socket.close()

    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.get_state()
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all(
                [np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.get_state()
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        self.tcp_socket.close()

    def go_home(self):

        self.move_joints(self.home_joint_config)

    def grasp(self, position, heightmap_rotation_angle, finger_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]), " with finger angle:{}".format(finger_angle*np.pi/180))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.3
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                         vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                    UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                    UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                    UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                    np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                              vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            if self.gripper_type == 'barrett':
                self.rotate_fingers(finger_angle)

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # Check if grasp is successful (by checking the height of the lifted object)
            time.sleep(1.5)  # object might drop from hand!
            object_positions = np.asarray(self.get_obj_positions())
            object_positions = object_positions[:, 2]
            grasped_object_height = np.max(object_positions)
            grasp_success = grasped_object_height > 0.2

            # Move the grasped object elsewhere
            if grasp_success:
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:, 2]
                grasped_object_ind = np.argmax(object_positions)
                grasped_object_handle = self.object_handles[grasped_object_ind]
                vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1,
                                           (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1),
                                           vrep.simx_opmode_blocking)
            else:
                pass  # TODO

        return grasp_success
