from collections import OrderedDict
import torchvision
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# HybridNet predict both grasping quality and grasping configuration
class HybridNet(nn.Module):

    def __init__(self, use_cuda):
        super(HybridNet, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.feature_dim = 2048

        self.num_rotations = 16

        # Construct network branches for grasping (Quality Map)
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Construct network branches for grasping (Configuration Map)
        self.confignet = nn.Sequential(OrderedDict([
            ('config-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('config-relu0', nn.ReLU(inplace=True)),
            ('config-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('config-norm1', nn.BatchNorm2d(64)),
            ('config-relu1', nn.ReLU(inplace=True)),
            ('config-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('config-norm2', nn.Tanh()),  # note the output is constrained to [0,1]
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' or 'config-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    interm_feat.append([interm_grasp_feat])

                    # print("feature shape:", interm_grasp_color_feat.shape)
                    # # feature shape: torch.Size([1, 1024, 20, 20])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_grasp_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_grasp_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([torch.clamp(nn.Upsample(scale_factor=16, mode='bilinear').forward(
                        F.grid_sample(self.confignet(interm_grasp_feat), flow_grid_after, mode='nearest')), -1, 1),
                        nn.Upsample(scale_factor=16, mode='bilinear').forward(
                            F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                          mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_grasp_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([torch.clamp(nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.confignet(interm_grasp_feat), flow_grid_after, mode='nearest')), -1, 1),
                nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                  mode='nearest'))])

            return self.output_prob, self.interm_feat


class HybridNet2(nn.Module):

    def __init__(self, use_cuda):
        super(HybridNet2, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.config_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.config_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.feature_dim = 2048

        self.num_rotations = 16

        # Construct network branches for grasping (Quality Map)
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Construct network branches for grasping (Configuration Map)
        self.confignet = nn.Sequential(OrderedDict([
            ('config-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('config-relu0', nn.ReLU(inplace=True)),
            ('config-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('config-norm1', nn.BatchNorm2d(64)),
            ('config-relu1', nn.ReLU(inplace=True)),
            ('config-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('config-norm2', nn.Tanh()),  # note the output is constrained to [0,1]
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' or 'config-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    interm_config_color_feat = self.config_color_trunk.features(rotate_color)
                    interm_config_depth_feat = self.config_depth_trunk.features(rotate_depth)
                    interm_config_feat = torch.cat((interm_config_color_feat, interm_config_depth_feat), dim=1)
                    interm_feat.append([interm_config_feat, interm_grasp_feat])

                    # print("feature shape:", interm_grasp_color_feat.shape)
                    # # feature shape: torch.Size([1, 1024, 20, 20])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_grasp_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_grasp_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                        F.grid_sample(self.confignet(interm_config_feat), flow_grid_after, mode='nearest')),
                        nn.Upsample(scale_factor=16, mode='bilinear').forward(
                            F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                          mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_grasp_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([torch.clamp(nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.confignet(interm_grasp_feat), flow_grid_after, mode='nearest')), -1, 1),
                nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                  mode='nearest'))])

            return self.output_prob, self.interm_feat


# Predict grasping quality for each grasping mode
class MultiQNet(nn.Module):

    def __init__(self, use_cuda):  # , snapshot=None
        super(MultiQNet, self).__init__()
        self.use_cuda = use_cuda
        self.num_rotations = 12
        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

        self.grasp_color_trunk_features = None
        self.grasp_depth_trunk_features = None

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    # st = time.time()
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(),
                                                     flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(),
                                                     flow_grid_before,
                                                     mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    # st_f = time.time()
                    interm_grasp_color_feat = self.grasp_color_trunk_features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk_features(rotate_depth)
                    # et_f = time.time()
                    # print('Execution time of feature extraction:', et_f - st_f, 'seconds')
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    interm_feat.append([interm_grasp_feat])

                    # print("feature shape:", interm_grasp_color_feat.shape)
                    # # feature shape: torch.Size([1, 1024, 20, 20])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_grasp_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_grasp_feat.data.size())
                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                        F.grid_sample(self.graspnet_1(interm_grasp_feat), flow_grid_after, mode='nearest')),
                        nn.Upsample(scale_factor=16, mode='bilinear').forward(
                            F.grid_sample(self.graspnet_2(interm_grasp_feat), flow_grid_after,
                                          mode='nearest'))])
                    # et = time.time()  # recording end time
                    # print('Execution time of forward passing:', et - st, 'seconds')
            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                 [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(),
                                             flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(),
                                             flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_grasp_color_feat = self.grasp_color_trunk_features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk_features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_grasp_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.graspnet_1(interm_grasp_feat), flow_grid_after, mode='nearest')),
                nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.graspnet_2(interm_grasp_feat), flow_grid_after,
                                  mode='nearest'))])

            return self.output_prob, self.interm_feat


class TeacherNet(MultiQNet):
    def __init__(self, use_cuda):  # , snapshot=None
        super(TeacherNet, self).__init__(use_cuda)

        # # Initialize network trunks with DenseNet pre-trained on ImageNet
        # self.grasp_color_trunk = torchvision.models.densenet.densenet161(pretrained=False)
        # self.grasp_depth_trunk = torchvision.models.densenet.densenet161(pretrained=False)
        # self.grasp_color_trunk_features = self.grasp_color_trunk.features
        # self.grasp_depth_trunk_features = self.grasp_color_trunk.features
        #
        # self.feature_dim = 4416  # DenseNet161

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        res_model = torchvision.models.resnet101(pretrained=True)

        self.grasp_1_color_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
        self.grasp_1_depth_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
        self.grasp_2_color_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
        self.grasp_2_depth_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])

        # self.feature_dim = 1024  # for ResNet-18 ResNet34
        self.feature_dim = 4096  # for ResNet-50, 101, 152

        # Construct multiple network branches for grasping (Quality Map)
        self.graspnet_1 = nn.Sequential(OrderedDict([
            ('grasp_1-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp_1-relu0', nn.ReLU(inplace=True)),
            ('grasp_1-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp_1-norm1', nn.BatchNorm2d(64)),
            ('grasp_1-relu1', nn.ReLU(inplace=True)),
            ('grasp_1-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        self.graspnet_2 = nn.Sequential(OrderedDict([
            ('grasp_2-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp_2-relu0', nn.ReLU(inplace=True)),
            ('grasp_2-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp_2-norm1', nn.BatchNorm2d(64)),
            ('grasp_2-relu1', nn.ReLU(inplace=True)),
            ('grasp_2-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp_' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    # st = time.time()
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(),
                                                     flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(),
                                                     flow_grid_before,
                                                     mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    # st_f = time.time()
                    interm_grasp_1_color_feat = self.grasp_1_color_trunk_features(rotate_color)
                    interm_grasp_1_depth_feat = self.grasp_1_depth_trunk_features(rotate_depth)
                    interm_grasp_2_color_feat = self.grasp_2_color_trunk_features(rotate_color)
                    interm_grasp_2_depth_feat = self.grasp_2_depth_trunk_features(rotate_depth)
                    interm_grasp_1_feat = torch.cat((interm_grasp_1_color_feat, interm_grasp_1_depth_feat), dim=1)
                    interm_grasp_2_feat = torch.cat((interm_grasp_2_color_feat, interm_grasp_2_depth_feat), dim=1)
                    interm_feat.append([interm_grasp_1_feat, interm_grasp_2_feat])
                    # et_f = time.time()
                    # print('Execution time of feature extraction:', et_f - st_f, 'seconds')

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_grasp_1_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_grasp_1_feat.data.size())
                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                        F.grid_sample(self.graspnet_1(interm_grasp_1_feat), flow_grid_after, mode='nearest')),
                        nn.Upsample(scale_factor=16, mode='bilinear').forward(
                            F.grid_sample(self.graspnet_2(interm_grasp_2_feat), flow_grid_after,
                                          mode='nearest'))])
                    # et = time.time()  # recording end time
                    # print('Execution time of forward passing:', et - st, 'seconds')
            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                 [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(),
                                             flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(),
                                             flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_grasp_1_color_feat = self.grasp_1_color_trunk_features(rotate_color)
            interm_grasp_1_depth_feat = self.grasp_1_depth_trunk_features(rotate_depth)
            interm_grasp_2_color_feat = self.grasp_2_color_trunk_features(rotate_color)
            interm_grasp_2_depth_feat = self.grasp_2_depth_trunk_features(rotate_depth)
            interm_grasp_1_feat = torch.cat((interm_grasp_1_color_feat, interm_grasp_1_depth_feat), dim=1)
            interm_grasp_2_feat = torch.cat((interm_grasp_2_color_feat, interm_grasp_2_depth_feat), dim=1)
            self.interm_feat.append([interm_grasp_1_feat, interm_grasp_2_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_grasp_1_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_grasp_1_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.graspnet_1(interm_grasp_1_feat), flow_grid_after, mode='nearest')),
                nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.graspnet_2(interm_grasp_2_feat), flow_grid_after,
                                  mode='nearest'))])

            return self.output_prob, self.interm_feat


class StudentNet(MultiQNet):
    def __init__(self, use_cuda):  # , snapshot=None
        super(StudentNet, self).__init__(use_cuda)

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        res_model = torchvision.models.resnet18(pretrained=True)

        self.grasp_color_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
        self.grasp_depth_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])

        self.feature_dim = 1024  # for ResNet-18
        # self.feature_dim = 4096  # for ResNet-50

        # Construct multiple network branches for grasping (Quality Map)
        self.graspnet_1 = nn.Sequential(OrderedDict([
            ('grasp_1-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp_1-relu0', nn.ReLU(inplace=True)),
            ('grasp_1-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp_1-norm1', nn.BatchNorm2d(64)),
            ('grasp_1-relu1', nn.ReLU(inplace=True)),
            ('grasp_1-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        self.graspnet_2 = nn.Sequential(OrderedDict([
            ('grasp_2-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('grasp_2-relu0', nn.ReLU(inplace=True)),
            ('grasp_2-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp_2-norm1', nn.BatchNorm2d(64)),
            ('grasp_2-relu1', nn.ReLU(inplace=True)),
            ('grasp_2-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp_' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

# class TeacherNets(nn.Module):
#
#     def __init__(self, use_cuda):  # , snapshot=None
#         super(TeacherNets, self).__init__()
#         self.use_cuda = use_cuda
#         self.num_rotations = 12
#         # Initialize output variable (for backprop)
#         self.interm_feat = []
#         self.output_prob = []
#
#         # Initialize network trunks with DenseNet pre-trained on ImageNet
#         res_model = torchvision.models.resnet101(pretrained=True)
#
#         self.grasp_color_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
#         self.grasp_depth_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])
#
#         # self.feature_dim = 1024  # for ResNet-18 ResNet34
#         self.feature_dim = 4096  # for ResNet-50, 101, 152
#
#         # Construct multiple network branches for grasping (Quality Map)
#         self.graspnet = nn.Sequential(OrderedDict([
#             ('grasp-norm0', nn.BatchNorm2d(self.feature_dim)),
#             ('grasp-relu0', nn.ReLU(inplace=True)),
#             ('grasp-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
#             ('grasp-norm1', nn.BatchNorm2d(64)),
#             ('grasp-relu1', nn.ReLU(inplace=True)),
#             ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
#         ]))
#
#         # Initialize network weights
#         for m in self.named_modules():
#             if 'grasp' in m[0]:
#                 if isinstance(m[1], nn.Conv2d):
#                     nn.init.kaiming_normal(m[1].weight.data)
#                 elif isinstance(m[1], nn.BatchNorm2d):
#                     m[1].weight.data.fill_(1)
#                     m[1].bias.data.zero_()
#
#     def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
#
#         if is_volatile:
#             with torch.no_grad():
#                 output_prob = []
#                 interm_feat = []
#
#                 # Apply rotations to images
#                 for rotate_idx in range(self.num_rotations):
#                     # st = time.time()
#                     rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))
#
#                     # Compute sample grid for rotation BEFORE neural network
#                     affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
#                                                     [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
#                     affine_mat_before.shape = (2, 3, 1)
#                     affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
#                     if self.use_cuda:
#                         flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
#                                                          input_color_data.size())
#                     else:
#                         flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
#                                                          input_color_data.size())
#
#                     # Rotate images clockwise
#                     if self.use_cuda:
#                         rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(),
#                                                      flow_grid_before,
#                                                      mode='nearest')
#                         rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(),
#                                                      flow_grid_before,
#                                                      mode='nearest')
#                     else:
#                         rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
#                                                      mode='nearest')
#                         rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
#                                                      mode='nearest')
#
#                     # Compute intermediate features
#                     # st_f = time.time()
#                     interm_grasp_color_feat = self.grasp_color_trunk_features(rotate_color)
#                     interm_grasp_depth_feat = self.grasp_depth_trunk_features(rotate_depth)
#                     # et_f = time.time()
#                     # print('Execution time of feature extraction:', et_f - st_f, 'seconds')
#                     interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
#                     interm_feat.append([interm_grasp_feat])
#
#                     # print("feature shape:", interm_grasp_color_feat.shape)
#                     # # feature shape: torch.Size([1, 1024, 20, 20])
#
#                     # Compute sample grid for rotation AFTER branches
#                     affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
#                                                    [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
#                     affine_mat_after.shape = (2, 3, 1)
#                     affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
#                     if self.use_cuda:
#                         flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
#                                                         interm_grasp_feat.data.size())
#                     else:
#                         flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
#                                                         interm_grasp_feat.data.size())
#                     # Forward pass through branches, undo rotation on output predictions, upsample results
#                     output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
#                         F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])
#                     # et = time.time()  # recording end time
#                     # print('Execution time of forward passing:', et - st, 'seconds')
#             return output_prob, interm_feat
#
#         else:
#             self.output_prob = []
#             self.interm_feat = []
#
#             # Apply rotations to intermediate features
#             # for rotate_idx in range(self.num_rotations):
#             rotate_idx = specific_rotation
#             rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))
#
#             # Compute sample grid for rotation BEFORE branches
#             affine_mat_before = np.asarray(
#                 [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
#                  [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
#             affine_mat_before.shape = (2, 3, 1)
#             affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
#             if self.use_cuda:
#                 flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
#                                                  input_color_data.size())
#             else:
#                 flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
#                                                  input_color_data.size())
#
#             # Rotate images clockwise
#             if self.use_cuda:
#                 rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(),
#                                              flow_grid_before,
#                                              mode='nearest')
#                 rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(),
#                                              flow_grid_before,
#                                              mode='nearest')
#             else:
#                 rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
#                                              mode='nearest')
#                 rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
#                                              mode='nearest')
#
#             # Compute intermediate features
#             interm_grasp_color_feat = self.grasp_color_trunk_features(rotate_color)
#             interm_grasp_depth_feat = self.grasp_depth_trunk_features(rotate_depth)
#             interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
#             self.interm_feat.append([interm_grasp_feat])
#
#             # Compute sample grid for rotation AFTER branches
#             affine_mat_after = np.asarray(
#                 [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
#             affine_mat_after.shape = (2, 3, 1)
#             affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
#             if self.use_cuda:
#                 flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
#                                                 interm_grasp_feat.data.size())
#             else:
#                 flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
#                                                 interm_grasp_feat.data.size())
#
#             # Forward pass through branches, undo rotation on output predictions, upsample results
#             self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
#                 F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])
#
#             return self.output_prob, self.interm_feat
