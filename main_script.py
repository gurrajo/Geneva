
import numpy as np
import cv2
import tag_detection_2
import requests
import time
import matplotlib.pyplot as plt


tag_type = 'aruco_4x4'
vidcap = cv2.VideoCapture('graphics/model_1_4k.mp4')
t = vidcap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3  # timestamp
success, image = vidcap.read()
count = 0
c_point = 0  # defines which corner to evaluate
geneva_object_0 = tag_detection_2.Geneva(tag_type)

while success:
    fname = f'graphics/cv/frame{count}.jpg'
    # cv2.imwrite(fname, image)  # save frame as JPEG file
    if success:
        geneva_object_0.detect_tags(image, t)
        t = vidcap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3  # timestamp
    if count == 1000 or count == 7800:  # shows marker detection
        geneva_object_0.draw_tags()
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

geneva_object_0.find_center()
geneva_object_0.find_angles()
geneva_object_0.corner_point_video()
geneva_object_0.normalize_signals()
geneva_object_0.theta_dot = geneva_object_0.calc_derivatives()
geneva_object_0.theta_bis = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot)
geneva_object_0.plot_signal(xlabel='t',ylabel='angle',title='angle all signals')
geneva_object_0.plot_signal(geneva_object_0.theta_dot,xlabel='t',ylabel='angular velocity',title='angular velocity all signals')
geneva_object_0.plot_signal(geneva_object_0.theta_bis,xlabel='t',ylabel='angular acceleration',title='angular acceleration all signals')

geneva_object_0.theta_smooth = geneva_object_0.smoothen_signal()

geneva_object_0.theta_dot_smooth = geneva_object_0.calc_derivatives(geneva_object_0.theta_smooth)
geneva_object_0.theta_bis_smooth = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot_smooth)
geneva_object_0.plot_signal(geneva_object_0.theta_smooth,xlabel='t',ylabel='angle',title='angle smoothened')
geneva_object_0.plot_signal(geneva_object_0.theta_dot_smooth,xlabel='t',ylabel='angular velocity',title='angular velocity smoothened')
geneva_object_0.plot_signal(geneva_object_0.theta_bis_smooth,xlabel='t',ylabel='angular acceleration',title='angular acceleration smoothened')

combined = geneva_object_0.combine_signal()
geneva_object_0.theta_dot_comb = geneva_object_0.calc_derivatives(combined)
geneva_object_0.theta_bis_comb = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot_comb)
geneva_object_0.plot_signal(combined,xlabel='t',ylabel='angle',title='angle combined')
geneva_object_0.plot_signal(geneva_object_0.theta_dot_comb,xlabel='t',ylabel='angular velocity',title='derivative of combined')
geneva_object_0.plot_signal(geneva_object_0.theta_bis_comb,xlabel='t',ylabel='angular acceleration',title='second derivative of combined')

plt.show()