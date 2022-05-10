import numpy as np
import cv2
import tag_detection_2
import matplotlib.pyplot as plt
import os
import re
import glob
test_nr = 8
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
geneva_object_0 = tag_detection_2.Geneva(tag_type, tag_id=0, filename=filename, rot_dir='CCW', test_nr=test_nr)
while success:
    if count == len(t):
        break
    if success:
        geneva_object_0.detect_tags(image, t[count])
    if count == 1000 or count == 7800:  # shows marker detection
        geneva_object_0.draw_tags()
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
rot_c_x, rot_c_y = geneva_object_0.find_center()
geneva_object_0.find_angles()
#geneva_object_0.corner_point_video()
geneva_object_0.normalize_signals()
geneva_object_0.theta_dot = geneva_object_0.calc_derivatives()
geneva_object_0.theta_bis = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot)

geneva_object_0.plot_signal(xlabel='time [sec]',ylabel='rad',title='angle corners', xlim=[0, 8.6])
geneva_object_0.plot_signal(geneva_object_0.theta_dot,xlabel='time [sec]',ylabel=r'$\frac{rad}{s}$',title='angular velocity corners', xlim=[0, 8.6], ylim=[-0.8,0.2])
geneva_object_0.plot_signal(geneva_object_0.theta_bis,xlabel='time [sec]',ylabel=r'$\frac{rad}{s^2}$',title='angular acceleration corners', xlim=[0, 8.6], ylim=[-12.75,12.75])

# geneva_object_0.theta_smooth = geneva_object_0.smoothen_signal()
#
# geneva_object_0.theta_dot_smooth = geneva_object_0.calc_derivatives(geneva_object_0.theta_smooth)
# geneva_object_0.theta_bis_smooth = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot_smooth)
# geneva_object_0.plot_signal(geneva_object_0.theta_smooth,xlabel='t',ylabel='rad',title='angle smoothened')
# geneva_object_0.plot_signal(geneva_object_0.theta_dot_smooth,xlabel='t',ylabel='rad/sec',title='angular velocity smoothened')
# geneva_object_0.plot_signal(geneva_object_0.theta_bis_smooth,xlabel='t',ylabel='rad/sec^2',title='angular acceleration smoothened')

# combined = geneva_object_0.combine_signal()
# geneva_object_0.theta_dot_comb = geneva_object_0.calc_derivatives(combined)
# geneva_object_0.theta_bis_comb = geneva_object_0.calc_derivatives(geneva_object_0.theta_dot_comb)
# geneva_object_0.plot_signal(combined,xlabel='t',ylabel='rad',title='angle combined')
# geneva_object_0.plot_signal(geneva_object_0.theta_dot_comb,xlabel='t',ylabel='rad/sec',title='derivative of combined')
# geneva_object_0.plot_signal(geneva_object_0.theta_bis_comb,xlabel='t',ylabel='rad/sec^2',title='second derivative of combined')

x_list, y_list, t_list = geneva_object_0.vibration_study((3.2, 3.8))

geneva_object_0.plot_signal(x_list, t=t_list, xlabel='time [sec]',ylabel='x',title='x values when stationary', legend=["Corner 0","Corner 1","Corner 2","Corner 3","Marker center"])
geneva_object_0.plot_signal(y_list, t=t_list, xlabel='time [sec]',ylabel='y',title='y values when stationary', legend=["Corner 0","Corner 1","Corner 2","Corner 3","Marker center"])

dx = geneva_object_0.x[30][0]-geneva_object_0.x[30][1]
dy = geneva_object_0.y[30][0]-geneva_object_0.y[30][1]
print(np.sqrt(dx**2+dy**2))

dx = geneva_object_0.x[30][1]-geneva_object_0.x[30][2]
dy = geneva_object_0.y[30][1]-geneva_object_0.y[30][2]
print(np.sqrt(dx**2+dy**2))

geneva_object_0.theta_mc_dot = geneva_object_0.calc_derivatives(geneva_object_0.theta_mc)
geneva_object_0.theta_mc_bis = geneva_object_0.calc_derivatives(geneva_object_0.theta_mc_dot)
geneva_object_0.plot_signal(geneva_object_0.theta_mc, xlabel='time [sec]', ylabel='rad', title='marker center angle')#, xlim=[0, 8.6])
geneva_object_0.plot_signal(geneva_object_0.theta_mc_dot, xlabel='time [sec]', ylabel=r'$\frac{rad}{s}$', title='marker center angular velocity')#, xlim=[0, 8.6], ylim=[-0.8,0.2])
geneva_object_0.plot_signal(geneva_object_0.theta_mc_bis, xlabel='time [sec]', ylabel=r'$\frac{rad}{s^2}$', title='marker center angular acceleration')#, xlim=[0, 8.6], ylim=[-12.75,12.75])

#geneva_object_0.corner_dist()
#geneva_object_0.data_to_text(geneva_object_0.theta_mc_bis, 'angular_acceleration_marker_center')
t_diff = np.diff(geneva_object_0.t)
t_diff = np.append(t_diff, t_diff[-1])
theta_error = 0.00095  # rad
theta_dot_error = theta_error/t_diff  # assumed constant time diff of 1/30 sec
theta_bis_error = theta_dot_error/t_diff
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(geneva_object_0.t, geneva_object_0.theta_mc_dot)
plt.fill_between(geneva_object_0.t, geneva_object_0.theta_mc_dot+theta_dot_error,
                 geneva_object_0.theta_mc_dot-theta_dot_error, color='red', alpha=0.3)
plt.xlim([0, 8.6])
plt.ylim([-0.8,0.2])
plt.title('angular velocity with error region')
plt.xlabel('time [sec]')
plt.ylabel(r'$\frac{rad}{s}$')
plt.savefig(f'graphics/plots/ang_vel_error.eps')

fig1, ax1 = plt.subplots()
xlim=[0, 8.6]
ylim=[-12.75,12.75]
plt.xlim([0, 8.6])
plt.ylim([-12.75,12.75])
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(geneva_object_0.t, geneva_object_0.theta_mc_bis)
plt.fill_between(geneva_object_0.t, geneva_object_0.theta_mc_bis+theta_bis_error,
                 geneva_object_0.theta_mc_bis-theta_bis_error, color='red', alpha=0.3)
plt.title('angular acceleration with error region')
plt.xlabel('time [sec]')
plt.ylabel(r'$\frac{rad}{s^2}$')
plt.savefig(f'graphics/plots/ang_acc_error.eps')

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
t_diff = np.diff(geneva_object_0.t)
plt.plot(range(len(t_diff)), t_diff)
plt.title('time difference between frames')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.savefig(f'graphics/plots/plot_time_diff.eps')

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
t_diff = np.diff(geneva_object_0.t)
plt.plot(range(len(t_diff)), t_diff)
plt.title('time difference between frames')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.xlim([20, 70])
plt.ylim([0.0332, 0.0338])
plt.savefig(f'graphics/plots/plot_time_diff_zoom.eps')

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(range(len(geneva_object_0.t)), geneva_object_0.t)
plt.title('time of each frame')
plt.xlabel('frame index')
plt.ylabel('seconds')
plt.savefig(f'graphics/plots/plot_time.eps')
plt.show()
