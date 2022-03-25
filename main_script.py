
import numpy as np
import cv2
import tag_detection_2

tag_type = 'apriltag_36h11'
vidcap = cv2.VideoCapture('graphics/test_vid.mp4')
success, image = vidcap.read()
count = 0
c_point = 0  # defines which corner to evaluate
geneva_object_0 = tag_detection_2.Geneva(c_point, tag_type)

while success:
    fname = f'graphics/cv/frame{count}.jpg'
    cv2.imwrite(fname, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    geneva_object_0.detect_tags(fname)

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
