import os
import numpy as np
import cv2
import time
import torch
from torch.autograd import Variable
from NN_models import TeacherNet, StudentNet, HybridNet, HybridNet2
from scipy import ndimage


# Plot prediction result
def get_prediction_vis(predictions, color_heightmap, best_pix_ind):
    canvas = None
    num_rotations = predictions.shape[0]
    for canvas_row in range(int(num_rotations / 4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row * 4 + canvas_col
            prediction_vis = predictions[rotate_idx, :, :].copy()
            # prediction_vis[prediction_vis < 0] = 0 # assume probability
            # prediction_vis[prediction_vis > 1] = 1 # assume probability
            prediction_vis = np.clip(prediction_vis, 0, 1)
            prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
            prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            if rotate_idx == best_pix_ind[0]:
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7,
                                            (255, 255, 255), 2)
            prediction_vis = ndimage.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations), reshape=False,
                                            order=0)
            background_image = ndimage.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False,
                                              order=0)
            prediction_vis = (
                    0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(
                np.uint8)
            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

    return canvas


# Basic trainer class
class Trainer(object):
    def __init__(self, force_cpu):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Initialize Loss
        # self.criterion = torch.nn.CrossEntropyLoss(reduce=False) # Cross entropy loss
        # self.criterion = torch.nn.MSELoss(reduce=False)  # MSE loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)  # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        self.iteration = 0
        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []

    # Preload execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
                                              delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert (color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:, :, c] = (input_depth_image[:, :, c] - image_mean[c]) / image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (
            input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        # st = time.time()
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        # et = time.time()  # recording end time
        # print('Execution time of inner forward passing:', et - st, 'seconds')

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions_1 = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                      int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                      int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
                grasp_predictions_2 = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,
                                      int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                      int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
            else:
                grasp_predictions_1 = np.concatenate((grasp_predictions_1,
                                                      output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                                      int(padding_width / 2):int(
                                                          color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                                      int(padding_width / 2):int(
                                                          color_heightmap_2x.shape[0] / 2 - padding_width / 2)]),
                                                     axis=0)
                grasp_predictions_2 = np.concatenate((grasp_predictions_2,
                                                      output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,
                                                      int(padding_width / 2):int(
                                                          color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                                      int(padding_width / 2):int(
                                                          color_heightmap_2x.shape[0] / 2 - padding_width / 2)]),
                                                     axis=0)

        return grasp_predictions_1, grasp_predictions_2, state_feat


class HybridTrainer(Trainer):
    def __init__(self, future_reward_discount, load_snapshot, snapshot_file, force_cpu):  # , snapshot=None
        super(HybridTrainer, self).__init__(force_cpu)

        # Fully convolutional network
        self.model = HybridNet2(use_cuda=self.use_cuda)
        self.future_reward_discount = future_reward_discount

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)

    def get_label_value(self, grasp_success, next_color_heightmap, next_depth_heightmap):
        # Compute current reward
        current_reward = grasp_success

        # Compute future reward
        if not grasp_success:
            future_reward = 0
        else:
            _, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap,
                                                                      next_depth_heightmap,
                                                                      is_volatile=True)
            future_reward = np.max(next_grasp_predictions)

            # # Experiment: use Q differences
            # push_predictions_difference = next_push_predictions - prev_push_predictions
            # grasp_predictions_difference = next_grasp_predictions - prev_grasp_predictions
            # future_reward = max(np.max(push_predictions_difference), np.max(grasp_predictions_difference))

        print('Current reward: %f' % current_reward)
        print('Future reward: %f' % future_reward)
        expected_reward = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (
            current_reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward

    # Compute labels and backpropagation

    def backprop(self, color_heightmap, depth_heightmap, best_pix_ind, label_value, grasp_config):

        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)

        # Compute labels for grasp quality
        label_q = np.zeros((1, 320, 320))
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label_q[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute quality mask
        label_weights = np.zeros((1, 320, 320))
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        config_area_size = 3
        config_area = np.zeros((224, 224))
        config_area[best_pix_ind[1] - config_area_size:best_pix_ind[1] + config_area_size,
        best_pix_ind[2] - config_area_size:best_pix_ind[2] + config_area_size] = 1

        # Compute labels for grasp primitive
        label_config = np.zeros((1, 320, 320))
        tmp_label = np.zeros((224, 224))
        tmp_label[config_area > 0] = grasp_config
        label_config[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute config mask
        config_mask = np.zeros((1, 320, 320))
        tmp_config_mask = np.zeros((224, 224))
        tmp_config_mask[config_area > 0] = 1
        config_mask[0, 48:(320 - 48), 48:(320 - 48)] = tmp_config_mask

        # Compute loss and backward pass
        self.optimizer.zero_grad()

        if label_value > 0:  # When successful, compute both quality and config loss

            # Do forward pass with specified rotation (to save gradients)
            config_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                             is_volatile=False,
                                                                             specific_rotation=best_pix_ind[0])
            if self.use_cuda:
                loss_config = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                             Variable(torch.from_numpy(label_config).float().cuda())) * Variable(
                    torch.from_numpy(config_mask).float().cuda(), requires_grad=False)
                loss_q = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                        Variable(torch.from_numpy(label_q).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss_config = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                             Variable(torch.from_numpy(label_config).float())) * Variable(
                    torch.from_numpy(config_mask).float(), requires_grad=False)
                loss_q = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                        Variable(torch.from_numpy(label_q).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)
            loss = loss_q.sum() + loss_config.sum()
            loss.backward()
            loss_q_value = loss_q.sum().cpu().data.numpy()
            loss_config_value = loss_config.sum().cpu().data.numpy()

            print('Q loss: %f;' % loss_q_value, 'config loss: %f;' % loss_config_value)
            self.optimizer.step()

        else:  # Only implement backpropagation on q net

            # Do forward pass with specified rotation (to save gradients)
            config_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                             is_volatile=False,
                                                                             specific_rotation=best_pix_ind[0])
            if self.use_cuda:
                loss_q = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                        Variable(torch.from_numpy(label_q).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss_q = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                        Variable(torch.from_numpy(label_q).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)
            loss = loss_q.sum()
            loss.backward()
            loss_q_value = loss_q.sum().cpu().data.numpy()
            loss_config_value = 0
            print('Q loss: %f' % loss_q_value)
            self.optimizer.step()
        return loss_q_value, loss_config_value


class MultiQTrainer(Trainer):
    def __init__(self, network_type, future_reward_discount, load_snapshot, snapshot_file,
                 force_cpu):  # , snapshot=None
        super(MultiQTrainer, self).__init__(force_cpu)

        self.grasp_mode_count = [0, 0]

        # Fully convolutional network
        if network_type == 'teacher':
            self.model = TeacherNet(use_cuda=self.use_cuda)
        elif network_type == 'student':
            self.model = StudentNet(use_cuda=self.use_cuda)
        else:
            raise ValueError("Network type error")

        self.future_reward_discount = future_reward_discount

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)

    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
                                              delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.grasp_mode_count[0] = np.count_nonzero(np.array(self.executed_action_log)[:, 0] == 1)
        self.grasp_mode_count[1] = np.count_nonzero(np.array(self.executed_action_log)[:, 0] == 1)

        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()

    def get_label_value(self, grasp_success, next_color_heightmap, next_depth_heightmap):

        # Compute current reward
        current_reward = grasp_success

        # Compute future reward
        if not grasp_success:
            future_reward = 0
        else:
            grasp_1_predictions, grasp_2_predictions, next_state_feat = self.forward(next_color_heightmap,
                                                                                     next_depth_heightmap,
                                                                                     is_volatile=True)
            future_reward = max(np.max(grasp_1_predictions), np.max(grasp_2_predictions))

        print('Current reward: %f' % current_reward)
        print('Future reward: %f' % future_reward)
        expected_reward = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (
            current_reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, best_pix_ind, label_value, grasp_type):

        # Compute labels for grasp quality
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros((1, 320, 320))
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        if grasp_type == 1:  # barrett three-finger grasp
            # Do forward pass with specified rotation (to save gradients)
            grasp_predictions_1, grasp_predictions_2, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                                is_volatile=False,
                                                                                specific_rotation=best_pix_ind[0])
            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute quality after rotating 120 degree
            one_third_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 3) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                                is_volatile=False,
                                                                                specific_rotation=one_third_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute grasping quality after rotating 240 degree
            sixty_rotate_idx = (best_pix_ind[0] + self.model.num_rotations * 2 / 3) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                                is_volatile=False,
                                                                                specific_rotation=sixty_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value / 3

        elif grasp_type == 2:

            # Do forward pass with specified rotation (to save gradients)
            grasp_predictions_1, grasp_predictions_2, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                                is_volatile=False,
                                                                                specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute grasping quality after rotating 180 degree
            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.forward(color_heightmap, depth_heightmap,
                                                                                is_volatile=False,
                                                                                specific_rotation=opposite_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value / 2

        else:
            print("Grasp type error. Undefined")

        print('Training loss: %f' % loss_value)
        self.optimizer.step()
        return loss_value


class TSTrainer(Trainer):
    def __init__(self, teacher_reward_discount, teacher_snapshot_file, load_snapshot, student_snapshot_file, force_cpu):
        super(TSTrainer, self).__init__(force_cpu)

        self.optimizer = None
        self.model = None
        self.grasp_mode_count = [0, 0]

        # Fully convolutional network
        self.teacher_model = TeacherNet(use_cuda=self.use_cuda)
        self.student_model = StudentNet(use_cuda=self.use_cuda)
        self.teacher_reward_discount = teacher_reward_discount

        # Load pretrained teacher model
        self.teacher_model.load_state_dict(torch.load(teacher_snapshot_file))
        # Load pretrained student model
        if load_snapshot:
            self.student_model.load_state_dict(torch.load(student_snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % student_snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.teacher_model = self.teacher_model.cuda()
            self.student_model = self.student_model.cuda()

        # Set model to training mode
        self.teacher_model.train()
        self.student_model.train()

        # Initialize optimizer
        self.teacher_optimizer = torch.optim.SGD(self.teacher_model.parameters(), lr=1e-4, momentum=0.9,
                                                 weight_decay=2e-5)
        self.student_optimizer = torch.optim.SGD(self.student_model.parameters(), lr=1e-4, momentum=0.9,
                                                 weight_decay=2e-5)

    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
                                              delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'real-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'teacher-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()

    def ts_forward(self, model_type, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
        if model_type == 'teacher':
            self.model = self.teacher_model
        elif model_type == 'student':
            self.model = self.student_model
        return self.forward(color_heightmap, depth_heightmap, is_volatile, specific_rotation)

    def get_label_value(self, grasp_success, best_pix_id, primitive_action, prev_color_heightmap, prev_depth_heightmap):
        # Compute current reward
        current_reward = grasp_success

        # Compute teacher reward
        teacher_grasp_1_predictions, teacher_grasp_2_predictions, next_state_feat = self.ts_forward('teacher',
                                                                                                    prev_color_heightmap,
                                                                                                    prev_depth_heightmap,
                                                                                                    is_volatile=True)

        if primitive_action == 'grasp_1':
            teacher_reward = teacher_grasp_1_predictions[best_pix_id]
        elif primitive_action == 'grasp_2':
            teacher_reward = teacher_grasp_2_predictions[best_pix_id]
        else:
            raise ValueError("invalid grasping mode!")

        print('Current reward: %f' % current_reward)
        print('Teacher reward: %f' % teacher_reward)
        expected_reward = current_reward + self.teacher_reward_discount * teacher_reward
        print('Expected reward: %f + %f x %f = %f' % (
            current_reward, self.teacher_reward_discount, teacher_reward, expected_reward))
        return expected_reward, current_reward

    def backprop(self, model_type, color_heightmap, depth_heightmap, best_pix_ind, label_value, grasp_type):

        if model_type == 'teacher':
            self.model = self.teacher_model
            self.optimizer = self.teacher_optimizer
        elif model_type == 'student':
            self.model = self.student_model
            self.optimizer = self.student_optimizer
        else:
            raise ValueError("Network model Error")

        # Compute labels for grasp quality
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros((1, 320, 320))
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        if grasp_type == 1:  # barrett three-finger grasp
            # Do forward pass with specified rotation (to save gradients)
            grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
                                                                                   depth_heightmap,
                                                                                   is_volatile=False,
                                                                                   specific_rotation=best_pix_ind[0])
            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute quality after rotating 120 degree
            one_third_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 3) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
                                                                                   depth_heightmap,
                                                                                   is_volatile=False,
                                                                                   specific_rotation=one_third_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute grasping quality after rotating 240 degree
            sixty_rotate_idx = (best_pix_ind[0] + self.model.num_rotations * 2 / 3) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
                                                                                   depth_heightmap,
                                                                                   is_volatile=False,
                                                                                   specific_rotation=sixty_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value / 3

        elif grasp_type == 2:

            # Do ts_forward pass with specified rotation (to save gradients)
            grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
                                                                                   depth_heightmap,
                                                                                   is_volatile=False,
                                                                                   specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            # Compute grasping quality after rotating 180 degree
            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

            grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
                                                                                   depth_heightmap,
                                                                                   is_volatile=False,
                                                                                   specific_rotation=opposite_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float().cuda())) * Variable(
                    torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
                                      Variable(torch.from_numpy(label).float())) * Variable(
                    torch.from_numpy(label_weights).float(), requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value / 2

        else:
            print("Grasp type error. Undefined")

        print('Training loss: %f' % loss_value)
        self.optimizer.step()
        return loss_value

    def student_backprop(self, color_heightmap, depth_heightmap, best_pix_ind, label_value, grasp_type):

        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros((1, 320, 320))
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        # Compute loss and backward pass
        self.student_optimizer.zero_grad()
        loss_value = 0

        if grasp_type == 1:  # barrett three-finger grasp
            # Compute quality after rotating 120 degree
            one_third_rotate_idx = (best_pix_ind[
                                        0] + self.student_model.num_rotations / 3) % self.student_model.num_rotations
            # Compute grasping quality after rotating 240 degree
            sixty_rotate_idx = (best_pix_ind[
                                    0] + self.student_model.num_rotations * 2 / 3) % self.student_model.num_rotations

            for rotation_index in [best_pix_ind[0], one_third_rotate_idx, sixty_rotate_idx]:
                # Do ts_forward pass with specified rotation (to save gradients)
                with torch.no_grad():
                    teacher_predictions_1, teacher_predictions_2, state_feat = self.ts_forward('teacher',
                                                                                               color_heightmap,
                                                                                               depth_heightmap,
                                                                                               is_volatile=False,
                                                                                               specific_rotation=rotation_index)
                student_predictions_1, student_predictions_2, state_feat = self.ts_forward('student',
                                                                                           color_heightmap,
                                                                                           depth_heightmap,
                                                                                           is_volatile=False,
                                                                                           specific_rotation=rotation_index)

                if self.use_cuda:
                    loss_student = self.criterion(self.student_model.output_prob[0][0].view(1, 320, 320),
                                                  Variable(torch.from_numpy(label).float().cuda())) * Variable(
                        torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                    loss_distillation = self.criterion(self.student_model.output_prob[0][0].view(1, 320, 320),
                                                       Variable(self.teacher_model.output_prob[0][0].view(1, 320, 320),
                                                                requires_grad=False))
                else:
                    loss_student = self.criterion(self.student_model.output_prob[0][0].view(1, 320, 320),
                                                  Variable(torch.from_numpy(label).float())) * Variable(
                        torch.from_numpy(label_weights).float(), requires_grad=False)
                    loss_distillation = self.criterion(self.student_model.output_prob[0][0].view(1, 320, 320),
                                                       Variable(self.teacher_model.output_prob[0][0].view(1, 320, 320),
                                                                requires_grad=False))
                loss = loss_student.sum() + 0.0001 * loss_distillation.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

        elif grasp_type == 2:

            # Compute grasping quality after rotating 180 degree
            opposite_rotate_idx = (best_pix_ind[
                                       0] + self.student_model.num_rotations / 2) % self.student_model.num_rotations

            for rotation_index in [best_pix_ind[0], opposite_rotate_idx]:

                # Do ts_forward pass with specified rotation (to save gradients)
                with torch.no_grad():
                    teacher_predictions_1, teacher_predictions_2, state_feat = self.ts_forward('teacher',
                                                                                               color_heightmap,
                                                                                               depth_heightmap,
                                                                                               is_volatile=False,
                                                                                               specific_rotation=rotation_index)
                student_predictions_1, student_predictions_2, state_feat = self.ts_forward('student',
                                                                                           color_heightmap,
                                                                                           depth_heightmap,
                                                                                           is_volatile=False,
                                                                                           specific_rotation=rotation_index)

                if self.use_cuda:
                    loss_student = self.criterion(self.student_model.output_prob[0][1].view(1, 320, 320),
                                                  Variable(torch.from_numpy(label).float().cuda())) * Variable(
                        torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                    loss_distillation = self.criterion(self.student_model.output_prob[0][1].view(1, 320, 320),
                                                       Variable(self.teacher_model.output_prob[0][1].view(1, 320, 320),
                                                                requires_grad=False))
                else:
                    loss_student = self.criterion(self.student_model.output_prob[0][1].view(1, 320, 320),
                                                  Variable(torch.from_numpy(label).float())) * Variable(
                        torch.from_numpy(label_weights).float(), requires_grad=False)
                    loss_distillation = self.criterion(self.student_model.output_prob[0][1].view(1, 320, 320),
                                                       Variable(self.teacher_model.output_prob[0][1].view(1, 320, 320),
                                                                requires_grad=False))
                loss = loss_student.sum() + 0.0001 * loss_distillation.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

        else:
            print("Grasp type error. Undefined")

        print('Training loss: %f' % loss_value)
        self.student_optimizer.step()
        return loss_value

    # def backprop(self, model_type, color_heightmap, depth_heightmap, best_pix_ind, label_value, grasp_type):
    #
    #     if model_type == 'teacher':
    #         self.model = self.teacher_model
    #         self.optimizer = self.teacher_optimizer
    #     elif model_type == 'student':
    #         self.model = self.student_model
    #         self.optimizer = self.student_optimizer
    #     else:
    #         raise ValueError("Network model Error")
    #
    #     # Compute labels for grasp quality
    #     label = np.zeros((1, 320, 320))
    #     action_area = np.zeros((224, 224))
    #     action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
    #     # blur_kernel = np.ones((5,5),np.float32)/25
    #     # action_area = cv2.filter2D(action_area, -1, blur_kernel)
    #     tmp_label = np.zeros((224, 224))
    #     tmp_label[action_area > 0] = label_value
    #     label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label
    #
    #     # Compute label mask
    #     label_weights = np.zeros((1, 320, 320))
    #     tmp_label_weights = np.zeros((224, 224))
    #     tmp_label_weights[action_area > 0] = 1
    #     label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights
    #
    #     # Compute loss and backward pass
    #     self.optimizer.zero_grad()
    #     loss_value = 0
    #
    #     if grasp_type == 1:  # barrett three-finger grasp
    #         # Compute quality after rotating 120 degree
    #         one_third_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 3) % self.model.num_rotations
    #         # Compute grasping quality after rotating 240 degree
    #         sixty_rotate_idx = (best_pix_ind[0] + self.model.num_rotations * 2 / 3) % self.model.num_rotations
    #
    #         for rotation_index in [best_pix_ind[0], one_third_rotate_idx, sixty_rotate_idx]:
    #             # Do forward pass with specified rotation (to save gradients)
    #             grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
    #                                                                                    depth_heightmap,
    #                                                                                    is_volatile=False,
    #                                                                                    specific_rotation=rotation_index)
    #             if self.use_cuda:
    #                 loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
    #                                       Variable(torch.from_numpy(label).float().cuda())) * Variable(
    #                     torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
    #             else:
    #                 loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
    #                                       Variable(torch.from_numpy(label).float())) * Variable(
    #                     torch.from_numpy(label_weights).float(), requires_grad=False)
    #
    #             loss = loss.sum()
    #             loss.backward()
    #             loss_value = loss.cpu().data.numpy()
    #
    #     elif grasp_type == 2:
    #
    #         # Compute grasping quality after rotating 180 degree
    #         opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations
    #
    #         for rotation_index in [best_pix_ind[0], opposite_rotate_idx]:
    #
    #             # Do ts_forward pass with specified rotation (to save gradients)
    #             grasp_predictions_1, grasp_predictions_2, state_feat = self.ts_forward(model_type, color_heightmap,
    #                                                                                    depth_heightmap,
    #                                                                                    is_volatile=False,
    #                                                                                    specific_rotation=rotation_index)
    #
    #             if self.use_cuda:
    #                 loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
    #                                       Variable(torch.from_numpy(label).float().cuda())) * Variable(
    #                     torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
    #             else:
    #                 loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320),
    #                                       Variable(torch.from_numpy(label).float())) * Variable(
    #                     torch.from_numpy(label_weights).float(), requires_grad=False)
    #             loss = loss.sum()
    #             loss.backward()
    #             loss_value = loss.cpu().data.numpy()
    #
    #     else:
    #         print("Grasp type error. Undefined")
    #
    #     print('Training loss: %f' % loss_value)
    #     self.optimizer.step()
    #     return loss_value
