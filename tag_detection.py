import cv2
import numpy as np
import csv


class Tag:
    """
    Image object containing tag information
    """
    def __init__(self, fname, tag_type):
        self.fname = fname
        self.image = cv2.imread(fname)
        self.tag_type = tag_type
        self.dict = self.get_dict()
        self.markers, self.corners, self.ids = self.detect_tags()  # necessary for rotation

    def get_dict(self):
        super().__init__()
        if self.tag_type == 'aruco_4x4':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        elif self.tag_type == 'aruco_original':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

        elif self.tag_type == 'apriltag_36h11':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
        else:
            used_dict = ""
            print("incorrect tag type")
        return used_dict

    def detect_tags(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)
        markers = []
        if ids is None:
            return markers, [], []
        for tag_id in range(len(ids)):
            for i in range(len(ids)):
                current_id = ids[i]
                if current_id == tag_id:
                    x = [corners[i, 0, 0, 0], corners[i, 0, 1, 0],
                         corners[i, 0, 2, 0], corners[i, 0, 3, 0]]

                    y = [corners[i, 0, 0, 1], corners[i, 0, 1, 1],
                         corners[i, 0, 2, 1], corners[i, 0, 3, 1]]
                    markers.append([x, y, tag_id])
        return markers, corners, ids

    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imwrite('graphics/rot.jpg', image)
        cv2.imshow('Tags', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
