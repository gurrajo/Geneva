import cv2
import numpy as np
import csv


class Geneva:
    """
    Image object containing tag information
    """
    def __init__(self, c_point, tag_type):
        self.c_point = c_point  # corner point, defines which marker corner to evaluate
        self.image = []
        self.tag_type = tag_type
        self.dict = self.get_dict()
        self.ids = []
        self.corners = []

        self.x = []  # x values of marker corners
        self.y = []  # y values of marker corners
        self.x_c = []  # x and y of center
        self.y_c = []
        self.theta = []

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

    def detect_tags(self, fname):
        self.image = cv2.imread(fname)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)
        if corners.any():
            points = corners[0][0]  # use only first marker (should only be one)
            self.x.append(points[self.c_point, 0])
            self.y.append(points[self.c_point, 1])
            self.corners = corners  # keeps only the latest corner for plotting purposes
            self.ids = ids  # keeps only the latest corner for plotting purposes
        else:
            print("failed to detect marker")




    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imshow('Tags', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_center(self):
        """find center of rotation for disc"""
        self.x_c = np.mean(self.x)
        self.y_c = np.mean(self.y)

    def find_angles(self):
        x = np.subtract(self.x, self.x_c)
        y = np.subtract(self.y, self.y_c)
        self.theta = np.arctan2(y, x)
