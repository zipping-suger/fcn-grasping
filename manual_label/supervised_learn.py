import os
import numpy as np
import cv2
import torch
import json
from logger import Logger
import warnings
from trainer import HybridTrainer, get_prediction_vis, MultiQTrainer

warnings.filterwarnings('ignore')  # CAREFULL!!! Supress all warnings


def supervised_train(trainer_type, model_save_name, data_dir='Train/heightmaps', save_dir='logs_supervised'):
    print('Test started')
    # Annotation data directory
    data_depth_dir = data_dir + '_depth'
    annotated_log_dir = data_dir + '_log_aug'  # augmented data annotation

    # Save prediction
    data_size = len(os.listdir(data_dir))
    print("Total data size:", data_size, "\n")

    # Initialize trainer
    if trainer_type == 'reconfig':
        trainer = HybridTrainer(future_reward_discount=0, load_snapshot=False, snapshot_file=None, force_cpu=False)
    elif trainer_type == 'multiQ':
        trainer = MultiQTrainer('teacher', future_reward_discount=0, load_snapshot=False, snapshot_file=None,
                                force_cpu=False)
    else:
        raise ValueError("Wrong trainer type")

    backprop_count = 0

    # Training loop
    for i in range(data_size):
        print("------------\n", "Processing image:", '%06d.0.color.png' % i)
        color_heightmap = cv2.imread(os.path.join(data_dir, '%06d.0.color.png' % (i)))
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_BGR2RGB)
        depth_heightmap = cv2.imread(os.path.join(data_depth_dir, '%06d.0.depth.png' % (i)), -1)
        depth_heightmap = depth_heightmap.astype(np.float32) / 100000

        # load json annotation
        json_file_name = '%06d.0.color.png.json' % (i)
        f = open(os.path.join(annotated_log_dir, json_file_name), "r")
        annotation_list = json.loads(f.read())

        if trainer_type == 'reconfig':
            for item in annotation_list:
                # load annotations
                pix = item['best_pix']
                label_value = item['success']
                rotation_index = (item['rotation_index'] + 4) % 16  # rotate 90 degrees
                hand_config = item['hand_config_index'] * 2 - 3  # normalize to [-1. 1]
                best_pix_ind = [rotation_index, pix[0], pix[1]]
                trainer.backprop(color_heightmap, depth_heightmap, best_pix_ind, label_value, hand_config)
                backprop_count += 1

        elif trainer_type == 'multiQ':
            for item in annotation_list:
                # load annotations
                pix = item['best_pix']
                label_value = item['success']
                rotation_index = (item['rotation_index'] + 3) % 12  # rotate 90 degrees
                hand_config = item['hand_config_index']
                best_pix_ind = [rotation_index, pix[0], pix[1]]
                trainer.backprop(color_heightmap, depth_heightmap, best_pix_ind, label_value, hand_config)
                backprop_count += 1
        else:
            raise ValueError("Wrong trainer type")

        print("----done----\n")
    print("Total backprop number:", backprop_count)

    torch.save(trainer.model.state_dict(),
               os.path.join(save_dir, 'snapshot_{}.pth'.format(model_save_name)))


def supervised_eval(trainer_type, model_save_name, data_dir='Train/heightmaps', save_dir='logs_supervised'):
    # Load trained net
    snapshot_file = os.path.join(save_dir, 'snapshot_{}.pth'.format(model_save_name))

    # Initialize trainer
    if trainer_type == 'reconfig':
        trainer = HybridTrainer(future_reward_discount=0, load_snapshot=False, snapshot_file=None, force_cpu=False)
    elif trainer_type == 'multiQ':
        trainer = MultiQTrainer('teacher', future_reward_discount=0, load_snapshot=False, snapshot_file=None,
                                force_cpu=False)
    else:
        raise ValueError("Wrong trainer type")

    trainer.model.load_state_dict(torch.load(snapshot_file))
    print('Pre-trained model snapshot loaded from: %s' % snapshot_file)

    # Visualize
    validate_data_dir = data_dir + '_val'
    validate_data_size = int(len(os.listdir(data_dir)) / 2)
    visualize_dir = os.path.join(save_dir, 'val_vis')
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    for j in range(validate_data_size):
        test_color_heightmap = cv2.imread(os.path.join(validate_data_dir, '%06d.0.color.png' % j))
        test_color_heightmap = cv2.cvtColor(test_color_heightmap, cv2.COLOR_BGR2RGB)
        test_depth_heightmap = cv2.imread(os.path.join(validate_data_dir, '%06d.0.depth.png' % j), -1)
        test_depth_heightmap = test_depth_heightmap.astype(np.float32) / 100000

        if trainer_type == 'reconfig':
            config_predictions, grasp_predictions, state_feat = trainer.forward(test_color_heightmap,
                                                                                test_depth_heightmap,
                                                                                is_volatile=True)
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)

            config_pred_vis = get_prediction_vis(config_predictions, test_color_heightmap, best_pix_ind)
            cv2.imwrite(os.path.join(visualize_dir, '%06d.0.config.png' % j), config_pred_vis)
            grasp_pred_vis = get_prediction_vis(grasp_predictions, test_color_heightmap, best_pix_ind)
            cv2.imwrite(os.path.join(visualize_dir, '%06d.0.grasp.png' % j), grasp_pred_vis)

        elif trainer_type == 'multiQ':
            grasp_predictions_1, grasp_predictions_2, state_feat = trainer.forward(test_color_heightmap,
                                                                                   test_depth_heightmap,
                                                                                   is_volatile=True)
            best_pix_ind_1 = np.unravel_index(np.argmax(grasp_predictions_1), grasp_predictions_1.shape)
            best_pix_ind_2 = np.unravel_index(np.argmax(grasp_predictions_2), grasp_predictions_2.shape)

            graps_pred_1_vis = get_prediction_vis(grasp_predictions_1, test_color_heightmap, best_pix_ind_1)
            cv2.imwrite(os.path.join(visualize_dir, '%06d.0.grasp_1.png' % j), graps_pred_1_vis)
            grasp_pred_2_vis = get_prediction_vis(grasp_predictions_2, test_color_heightmap, best_pix_ind_2)
            cv2.imwrite(os.path.join(visualize_dir, '%06d.0.grasp_2.png' % j), grasp_pred_2_vis)


if __name__ == '__main__':
    # supervised_train(trainer_type='multiQ',model_save_name='multiQtest')
    supervised_eval(trainer_type='multiQ', model_save_name='multiQtest')
