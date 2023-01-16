import os
import numpy as np
import json
import cv2
import copy


def check_legal(best_pix):
    x_inbound = 0 <= best_pix[0] <= 223
    y_inbound = 0 <= best_pix[1] <= 223

    return x_inbound and y_inbound


if __name__ == '__main__':
    # Annotation data directory
    data_dir = 'manual_data/YCB_sim/height_map'
    annotated_log_dir = data_dir + '_log'
    aug_annotated_log_dir = data_dir + '_log_aug'

    data_size = len(os.listdir(data_dir))
    print("Total data size:", data_size, "\n")

    success_multiple = 4
    fail_multiple = 3

    success_counter = 0
    fail_counter = 0

    for i in range(data_size):
        # load json annotation
        json_file_name = '%06d.0.color.png.json' % i
        f = open(os.path.join(annotated_log_dir, json_file_name), "r")
        annotation_list = json.loads(f.read())
        output_annotation_list = copy.deepcopy(annotation_list)

        for item in annotation_list:
            success = item['success']
            if success:  # Successful Grasp
                i = 0
                success_counter += 1
                while i < success_multiple:
                    item_copy = copy.deepcopy(item)
                    # jittering x,y
                    item_copy["best_pix"][0] = item_copy["best_pix"][0] + int(np.random.randint(-3, 3, 1)[0])
                    item_copy["best_pix"][1] = item_copy["best_pix"][1] + int(np.random.randint(-3, 3, 1)[0])
                    if check_legal(item_copy["best_pix"]):
                        output_annotation_list.append(item_copy)
                        i += 1
                        success_counter += 1
            else:
                i = 0
                fail_counter += 1
                while i < fail_multiple:
                    item_copy = copy.deepcopy(item)
                    # jittering x,y
                    item_copy["best_pix"][0] = item_copy["best_pix"][0] + int(np.random.randint(-3, 3, 1)[0])
                    item_copy["best_pix"][1] = item_copy["best_pix"][1] + int(np.random.randint(-3, 3, 1)[0])
                    if check_legal(item_copy["best_pix"]):
                        output_annotation_list.append(item_copy)
                        i += 1
                        fail_counter += 1

        with open(os.path.join(aug_annotated_log_dir, json_file_name), "w") as jsonFile:
            json.dump(output_annotation_list, jsonFile)

    print("Total success:", success_counter)
    print("Total failure:", fail_counter)
