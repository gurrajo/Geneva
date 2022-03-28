import cv2
import numpy as np
import matplotlib.pyplot as plt


class Geneva:
    """
    Object representing a geneva drive
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
        self.theta_dot = []
        self.theta_bis = []
        self.t = []
        self.t_increment = 1/30  # frame rate
        self.theta_new = []
        self.t_new = []

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

        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)

        if corners.any():
            points = corners[0][0]  # use only first marker (should only be one)
            self.x.append(points[self.c_point, 0])
            self.y.append(points[self.c_point, 1])
            self.corners = corners  # keeps only the latest corner for plotting purposes
            self.ids = ids  # keeps only the latest corner for plotting purposes
            if self.t:
                self.t.append(self.t[-1] + self.t_increment)
                self.t_increment = 1/30
            else:
                self.t = [0.0]
            self.image.append(image)
        else:
            self.t_increment += 1/30
            print("failed to detect marker")

    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imshow('Tags', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_center(self):
        """find center of rotation for disc
        assumes a full rotation has been done"""

        self.x_c = np.mean((np.max(self.x), np.min(self.x)))
        self.y_c = np.mean((np.max(self.y), np.min(self.y)))

    def find_angles(self):
        x = np.subtract(self.x, self.x_c)
        y = np.subtract(self.y, self.y_c)
        self.theta = np.arctan2(y, x)
        for i, ang in enumerate(self.theta):
            if ang < 0:
                self.theta[i:-1] += np.pi*2
                break
        self.theta = np.delete(self.theta, -1)
        del self.t[-1]

    def corner_point_video(self):
        for i, p in enumerate(self.x):
            cv2.circle(self.image[i], (p, self.y[i]), 10, (255, 0, 0))
            cv2.circle(self.image[i], (self.x_c, self.y_c), 10, (0, 0, 0))
        height, width, layers = self.image[0].shape
        size = (width, height)
        out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.image)):
            out.write(self.image[i])
        out.release()

    def smoothen_signal(self):
        """average 3 data points into 1"""
        points = 3
        i = 0
        self.theta_new = []
        self.t_new = []
        while i < len(self.theta)/3:
            self.theta_new.append(np.mean(self.theta[i*points:(i+1)*points-1]))
            self.t_new.append(self.t[points*i])
            i += 1

    def calc_theta_derivatives(self):
        self.theta_dot = np.gradient(self.theta, self.t)
        self.theta_bis = np.gradient(self.theta_dot, self.t)

    def plot_derivatives(self):
        plt.plot(self.t, self.theta_dot)
        plt.show()
        plt.plot(self.t, self.theta_bis)
        plt.show()

    def plot_angles(self):
        print(len(self.theta))
        print(len(self.t))
        plt.plot(self.t, self.theta)
        plt.show()

    def plot_smooth(self):
        t_new = self.t_new
        theta_dot_new = np.gradient(self.theta_new, self.t_new)
        theta_bis_new = np.gradient(theta_dot_new, self.t_new)
        plt.plot(t_new, theta_dot_new)
        plt.show()
        plt.plot(t_new, theta_bis_new)
        plt.show()
