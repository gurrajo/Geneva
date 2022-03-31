
import numpy as np
import cv2
import tag_detection_2
import matplotlib.pyplot as plt

tag_type = 'aruco_4x4'
vidcap = cv2.VideoCapture('graphics/model_1_4k_nolight.mp4')
success, image = vidcap.read()
count = 0
c_point = 2  # defines which corner to evaluate
geneva_object_0 = tag_detection_2.Geneva(c_point, tag_type)

while success:
    fname = f'graphics/cv/frame{count}.jpg'
    # cv2.imwrite(fname, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    if success:
        geneva_object_0.detect_tags(image)
    if count == 305: # shows marker detection
        geneva_object_0.draw_tags()
geneva_object_0.find_center()
geneva_object_0.find_angles()
geneva_object_0.corner_point_video()
geneva_object_0.plot_angles()
geneva_object_0.calc_theta_derivatives()
geneva_object_0.plot_derivatives()
geneva_object_0.smoothen_signal()
geneva_object_0.plot_smooth()
# fname = f'graphics/test/0.0_1.5.jpg'
# c_point = 0  # defines which corner to evaluate
# geneva_object_0 = tag_detection_2.Geneva(c_point, tag_type)
# geneva_object_0.detect_tags(fname)
#
# print(geneva_object_0.x)
#
# c_point = 1  # defines which corner to evaluate
# geneva_object_1 = tag_detection_2.Geneva(c_point, tag_type)
# geneva_object_1.detect_tags(fname)
#
# print(geneva_object_1.x)
