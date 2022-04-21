
import numpy as np
import cv2
import tag_detection_2
import requests
import time
import matplotlib.pyplot as plt
from subprocess import check_output
import re

pts = str(check_output('ffmpeg -i graphics\model_1_4k.mp4 -vf select="eq(pict_type\,I)" -an -vsync 0  keyframes%03d.jpg -loglevel debug 2>&1 |findstr select:1  ',shell=True),'utf-8')  # ffmpeg call
pts_I = [float(i) for i in re.findall(r"\bpts:(\d+\.\d)", pts)] # Find pattern that starts with "pts:"
n_I = [float(i) for i in re.findall(r"\bn:(\d+\.\d)", pts)] # Find pattern that starts with "n:"
n_I = [int(i) for i in n_I]

pts = str(check_output('ffmpeg -i graphics\model_1_4k.mp4 -vf select="eq(pict_type\,P)" -an -vsync 0  keyframes%03d.jpg -loglevel debug 2>&1 |findstr select:1  ',shell=True),'utf-8')  # ffmpeg call
pts_P = [float(i) for i in re.findall(r"\bpts:(\d+\.\d)", pts)] # Find pattern that starts with "pts:"
n_P = [float(i) for i in re.findall(r"\bn:(\d+\.\d)", pts)] # Find pattern that starts with "n:"
n_P = [int(i) for i in n_P]

t = np.zeros((len(n_P) + len(n_I),1))

for i, n in enumerate(n_P):
    t[n] = pts_P[i]*1E-5  # seconds

for i, n in enumerate(n_I):
    t[n] = pts_I[i]*1E-5  # seconds

tag_type = 'aruco_4x4'
vidcap = cv2.VideoCapture('graphics/model_1_4k.mp4')
success, image = vidcap.read()
count = 0
geneva_object_0 = tag_detection_2.Geneva(tag_type, tag_id=0)
geneva_object_1 = tag_detection_2.Geneva(tag_type, tag_id=1)
geneva_object_2 = tag_detection_2.Geneva(tag_type, tag_id=2)

while success:
    fname = f'graphics/cv/frame{count}.jpg'
    # cv2.imwrite(fname, image)  # save frame as JPEG file
    if success:
        geneva_object_0.detect_tags(image, t[count][0])
        geneva_object_1.detect_tags(image, t[count][0])
        geneva_object_2.detect_tags(image, t[count][0])
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
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
t_diff = np.diff(geneva_object_0.t)
plt.plot(range(len(t_diff)), t_diff)
plt.title('time-difference between frames')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.savefig(f'graphics/plots/plot_time_diff.png')

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
t_diff = np.diff(geneva_object_0.t)
plt.plot(range(len(t_diff)), t_diff)
plt.title('time-difference between frames')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.xlim([28, 37])
plt.ylim([0.03, 0.0305])
plt.savefig(f'graphics/plots/plot_time_diff_zoom.png')

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(range(len(geneva_object_0.t)), geneva_object_0.t)
plt.title('time of each frame')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.savefig(f'graphics/plots/plot_time.png')
plt.show()
