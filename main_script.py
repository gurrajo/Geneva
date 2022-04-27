import numpy as np
import cv2
import tag_detection_2
import matplotlib.pyplot as plt
import os
import re
import glob
test_nr = 4
filename = f'model_test{test_nr}.mp4'
first_time = False  # False if time info text file exists
if first_time:
    os.system(f'ffmpeg -hide_banner -i graphics/{filename} -filter:v showinfo -y > graphics/time/{filename}info.txt 2>&1 graphics\junk\output%d.png')  # write text file with video metadata
    # remove junk files
    files = glob.glob('graphics/junk/*')
    for f in files:
        os.remove(f)

file = open(f'graphics/time/{filename}info.txt', mode='r')
t = [0]
for line in file:
    pts_P = re.findall(r"\spts_time:(\d+\.\d+)", line) # Find pattern that starts with "pts_time:"
    if pts_P:
        t.append(float(pts_P[0]))
print(len(t))

file.close()  # close the file
tag_type = 'aruco_4x4'

vidcap = cv2.VideoCapture(f'graphics/{filename}')
success, image = vidcap.read()
count = 0
geneva_object_0 = tag_detection_2.Geneva(tag_type, tag_id=2, filename=filename, rot_dir='CW', test_nr=test_nr)
geneva_object_1 = tag_detection_2.Geneva(tag_type, tag_id=1, filename=filename)
geneva_object_2 = tag_detection_2.Geneva(tag_type, tag_id=2, filename=filename)
while success:
    if count == len(t):
        break
    if success:
        geneva_object_0.detect_tags(image, t[count])
        geneva_object_1.detect_tags(image, t[count])
        geneva_object_2.detect_tags(image, t[count])
    if count == 1000 or count == 7800:  # shows marker detection
        geneva_object_0.draw_tags()
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
rot_c_x, rot_c_y = geneva_object_0.find_center()
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

x_list, y_list, t_list = geneva_object_0.vibration_study((1, 3))

geneva_object_0.plot_signal(x_list, t=t_list, xlabel='t',ylabel='x',title='vibration study')
geneva_object_0.plot_signal(y_list, t=t_list, xlabel='t',ylabel='y',title='vibration study')


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
