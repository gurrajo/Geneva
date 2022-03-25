
import numpy as np
import cv2

tag_type = 'aruco_4x4'
#vidcap = cv2.VideoCapture('')
#success, image = vidcap.read()
count = 0
center = [0, 0]  # center of rotation(x,y)

# while success:
#     fname = f'graphics/cv/frame{count}'
#     cv2.imwrite(fname, image)  # save frame as JPEG file
#     success, image = vidcap.read()
#     print('Read a new frame: ', success)
#     count += 1
#     new_image = tag_detection.Tag(fname, tag_type)
#     corners = new_image.corners

fname = f'graphics/cv/frame{count}'
new_image = tag_detection.Tag(fname, tag_type)
corners = new_image.corners

