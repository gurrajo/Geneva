import cv2
import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy import signal


class Geneva:
    """
    Object representing a geneva drive
    """
    def __init__(self, tag_type, tag_id, filename):
        self.filename = filename
        self.image = []
        self.tag_id = tag_id
        self.tag_type = tag_type
        self.dict = self.get_dict()
        self.ids = []
        self.corners = []
        self.rot_dir = 'CCW'  # or CW
        self.frame_remove = 7  # number of end frames to remove

        self.x = []  # x values of marker corners
        self.y = []  # y values of marker corners
        self.x_c = []  # x and y of center
        self.y_c = []
        self.theta = []
        self.theta_dot = []
        self.theta_bis = []
        self.t = []
        self.theta_norm = []

        self.plot_count = 0

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

    def detect_tags(self, image, t):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)
        if bool(list(ids)):
            for i, tag_id in enumerate(ids):
                if tag_id == self.tag_id:
                    points = corners[i][0]
                    self.x.append(points[:, 0])
                    self.y.append(points[:, 1])
                    self.corners = corners  # keeps only the latest corner for plotting purposes
                    self.ids = ids  # keeps only the latest ids for plotting purposes
                    self.t.append(t)
                    self.image.append(image)
        else:
            print("failed to detect marker")

    def normalize_signals(self):
        ref_point = 10  # reference frame
        self.theta_norm = np.subtract(self.theta, self.theta[ref_point, :])

    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image[-1], (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imshow('Tags', frame)
        cv2.imwrite('graphics/tag_det.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_center(self):
        """"""
        def get_intersect(a1, a2, b1, b2):
            """
            Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
            a1: [x, y] a point on the first line
            a2: [x, y] another point on the first line
            b1: [x, y] a point on the second line
            b2: [x, y] another point on the second line
            """
            s = np.vstack([a1, a2, b1, b2])  # s for stacked
            h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
            l1 = np.cross(h[0], h[1])  # get first line
            l2 = np.cross(h[2], h[3])  # get second line
            x, y, z = np.cross(l1, l2)  # point of intersection
            if z == 0:  # lines are parallel
                return float('inf'), float('inf')
            return int(x / z), int(y / z)

        x_1 = self.x[int((len(self.x)/4))][0]
        y_1 = self.y[int((len(self.x)/4))][0]
        x_2 = self.x[int((len(self.x)*3/4))][0]  # points not to close to one another
        y_2 = self.y[int((len(self.x)*3/4))][0]

        def f_1(x):
            return (y_1 + y_2)/2 + (x_2-x_1)/(y_2 - y_1)*(x_1+x_2)/2 - (x_2-x_1)/(y_2-y_1)*x

        a_1 = [0, f_1(0)]
        a_2 = [100, f_1(100)]

        image = cv2.line(self.image[0], (0, int(f_1(0))), (4000, int(f_1(4000))), (0, 0, 0))
        image = cv2.circle(image, (x_1, y_1), 10, (0, 0, 0))
        image = cv2.circle(image, (x_2, y_2), 10, (0, 0, 0))

        x_1 = self.x[int((len(self.x)/4))][1]
        y_1 = self.y[int((len(self.x)/4))][1]
        x_2 = self.x[int((len(self.x)*3/4))][1]
        y_2 = self.y[int((len(self.x)*3/4))][1]

        image = cv2.circle(image, (x_2, y_2), 10, (0, 0, 0))


        def f_2(x):
            return (y_1 + y_2)/2 + (x_2-x_1)/(y_2 - y_1)*(x_1+x_2)/2 - (x_2-x_1)/(y_2-y_1)*x

        b_1 = [0, f_2(0)]
        b_2 = [100, f_2(100)]
        (self.x_c, self.y_c) = get_intersect(a_1, a_2, b_1, b_2)
        image = cv2.line(image, (0, int(f_2(0))), (4000, int(f_2(4000))), (0, 0, 0))
        #cv2.imshow('Intersection', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return self.x_c, self.y_c

    def find_angles(self):
        x = np.subtract(self.x, self.x_c)
        y = np.subtract(self.y, self.y_c)

        self.theta = np.arctan2(y, x)

        for i in range(4):
            for j in range(len(self.theta)):
                ang = self.theta[j, i]
                if self.rot_dir == 'CW' and ang < 0 and self.theta[j-1,i] > 0:
                    self.theta[j:-1, i] += np.pi*2
                    break
                elif self.rot_dir == 'CCW' and ang > 0 and self.theta[j-1,i] < 0:
                    self.theta[j:-1, i] -= np.pi * 2
                    break

        for i in range(self.frame_remove):  # remove unwanted last values
            self.theta = np.delete(self.theta, [-1], 0)
            del self.t[-1]

    def corner_point_video(self):
        for i, p in enumerate(self.x):
            for j in range(4):
                cv2.circle(self.image[i], (p[j], self.y[i][j]), 5, (240, 240, 0),2)
            cv2.circle(self.image[i], (self.x_c, self.y_c), 10, (0, 255, 255))
        height, width, layers = self.image[0].shape
        size = (width, height)
        out = cv2.VideoWriter(f'graphics/vids/{self.filename}_{self.tag_id}_.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.image)):
            out.write(self.image[i])
        out.release()

    def calc_derivatives(self, sig=None):
        if sig is None:
            sig = self.theta_norm
        shape = np.shape(sig)
        derivative = np.zeros(shape)
        if len(shape) == 1:
            derivative = np.gradient(sig, self.t)
        else:
            for i in range(shape[1]):
                derivative[:, i] = np.gradient(sig[:, i], self.t)

        return derivative

    def smoothen_signal(self, sig=None):
        if sig is None:
            sig = self.theta_norm
        b, a = signal.butter(2, 0.3)  # coefficients worth looking at
        shape = np.shape(sig)
        smooth = np.zeros(shape)
        for i in range(shape[1]):
            smooth[:, i] = signal.filtfilt(b, a, sig[:, i])
        return smooth

    def combine_signal(self, sig=None):
        if sig is None:
            sig = self.theta_norm
        combined = np.average(sig, 1)
        return combined

    def plot_signal(self, sig=None, title="", xlabel="", ylabel=""):
        if sig is None:
            sig = self.theta_norm
        fig1, ax1 = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)
        plt.plot(self.t, sig)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f'graphics/plots/plot{self.plot_count}.png')
        self. plot_count += 1

