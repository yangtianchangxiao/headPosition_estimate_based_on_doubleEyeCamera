import cv2
import numpy as np

left_camera_matrix = np.array([[745.7529, 0.1488, 344.5329],
                               [0, 750.1008, 253.2383],
                               [0., 0., 1.]])
left_distortion = np.array([[0.2232, -1.2455, -0.0014, 0.0023, -0.2597]])

right_camera_matrix = np.array([[734.8314, 1.0615, 336.2630],
                                [0, 738.2798, 267.4528],
                                [0, 0, 1.0000]])

right_distortion = np.array([[0.3381, -2.4884, 0.0022, 0.0025, 4.6913]])

R = np.matrix([
    [1.0000, 0.0022, 0.0022],
    [-0.0022, 1.0000, 0.0088],
    [-0.0022, -0.0088, 1.0000],
])

# print(R)

T = np.array([-18.0133, 1.0184, 0.9606])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)