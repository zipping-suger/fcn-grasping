# import the required library
import cv2
import numpy as np
import json
import os

# Data to be written
annotation = list()


def plot_line(centre_point, rotate_theta, radius, color=(0, 255, 255)):
    start_point = centre_point + np.array(
        [np.floor(radius * np.cos(rotate_theta)), np.floor(radius * np.sin(rotate_theta))])
    end_point = centre_point - np.array(
        [np.floor(radius * np.cos(rotate_theta)), np.floor(radius * np.sin(rotate_theta))])
    cv2.line(img, start_point.astype(int), end_point.astype(int), color, thickness=1)


def plot_Y(centre_point, rotate_theta, radius, color=(0, 0, 255)):
    start_point = centre_point
    for i in range(3):
        end_point = centre_point + np.array(
            [np.floor(radius * np.cos(rotate_theta + 2*np.pi * i / 3)),
             np.floor(radius * np.sin(rotate_theta + 2*np.pi * i / 3))])
        cv2.line(img, start_point.astype(int), end_point.astype(int), color, thickness=1)


# define a function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    line_length = 20  # pixel
    num_rotations = 16  # rotation number
    global annotation

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')
        # put coordinates as text on the image
        cv2.putText(img, f'({x},{y})', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # draw point on the image
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)

        # TODO interactive annotation
        success = bool(input("Success? Yes:Any Key; No: Enter\n"))
        hand_config_index = int(input("Input grasp configuration index\n"))
        rotation_index = int(input("Input principle rotation_index\n")) % 16

        rotate_theta = np.radians(rotation_index * (360 / num_rotations))
        if hand_config_index == 1:
            plot_line(np.array([x, y]), rotate_theta, line_length, color=(0, 255, 255))
        elif hand_config_index == 2:
            plot_Y(np.array([x, y]), rotate_theta, line_length, color=(255, 255, 0))

        annotation.append({
            "name": img_name,
            "best_pix": (x, y),
            "success": success,
            "rotation_index": rotation_index,
            "hand_config_index": hand_config_index
        })

        print("------------------")


if __name__ == '__main__':

    # Annotation data directory
    data_dir = 'manual_data/YCB_sim/height_map'

    annotated_view_dir = data_dir + '_view'
    if not os.path.exists(annotated_view_dir):
        os.makedirs(annotated_view_dir)

    annotated_log_dir = data_dir + '_log'
    if not os.path.exists(annotated_log_dir):
        os.makedirs(annotated_log_dir)

    img_list = os.listdir(data_dir)
    print(img_list)

    for img_name in img_list:

        print('Processing:', img_name)

        # read the input image
        img = cv2.imread(os.path.join(data_dir, img_name))
        # create a window
        cv2.namedWindow('Point Coordinates')

        # bind the callback function to window
        cv2.setMouseCallback('Point Coordinates', click_event)

        # display the image
        while True:
            cv2.imshow('Point Coordinates', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:  # press space to save the annotation
                break
            elif k == 27:  # press esc to re-annotate the image
                img = cv2.imread(os.path.join(data_dir, img_name))
                annotation = list()

        with open(os.path.join(annotated_log_dir, img_name + ".json"), "w") as outfile:
            json.dump(annotation, outfile)
        annotation = list()  # Clean the memory!!!

        cv2.imwrite(os.path.join(annotated_view_dir, img_name), img)
        # cv2.destroyAllWindows()

    print("All annotated.\n")


